import math
import time
from copy import deepcopy
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, init
from torch.nn.modules.utils import _pair
from torch.nn.functional import relu, avg_pool2d
from torch.optim.lr_scheduler import StepLR

from models.utils.incremental_model import IncrementalModel

import numpy as np

'''
Co2In
'''


class BMKPV2(IncrementalModel):
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, args):
        super(BMKPV2, self).__init__(args)

        self.epochs = args.n_epochs
        self.retrain_epochs = args.retrain_epochs
        self.learning_rate = args.lr
        self.threshold_first = args.threshold_first
        self.threshold = args.threshold
        self.lambd = args.lambd

        self.net = None
        self.loss = F.cross_entropy

        self.feature_list = []
        self.param_weights = []
        self.bn_stats = []

        self.current_task = -1

        self.resnet_size = 0
        self.weight_size = 0
        self.basis_size = 0

    def begin_il(self, dataset):
        self.cpt = int(dataset.nc / dataset.nt)
        self.t_c_arr = dataset.t_c_arr
        self.eye = torch.tril(torch.ones((dataset.nc, dataset.nc))).bool().to(self.device)

        if self.args.dataset == 'seq-tinyimg' or self.args.dataset == 'seq-imagenet':
            self.img_size = 64
        else:
            self.img_size = 32

        nc = dataset.nc
        if self.args.is_rotate:
            nc *= 4

        if self.args.backbone == 'None':
            self.net = ResNet18(nc, nf=self.args.nf, nt=dataset.nt, img_size=self.img_size,
                            is_bn_stats=self.args.is_bn_stats).to(self.device)
        elif self.args.backbone == 'resnet34':
            self.net = ResNet34(nc, nf=self.args.nf, nt=dataset.nt, img_size=self.img_size,
                            is_bn_stats=self.args.is_bn_stats).to(self.device)
        elif self.args.backbone == 'lenet':
            self.net = LeNet(dataset.nt, self.cpt, nf=self.args.nf).to(self.device)

    def train_task(self, dataset, train_loader):
        self.current_task += 1

        e_sample, e_label = [], []
        for step, data in enumerate(train_loader):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            e_sample.append(inputs)
            e_label.append(labels)
            if step > (self.args.example_num / self.args.batch_size): break
        e_sample = torch.cat(e_sample)

        # train
        start_time = time.time()
        self.net.set_mode('train')
        self.train_(train_loader, epochs=self.epochs)
        train_time = start_time - time.time()

        # decompose
        start_time = time.time()
        self.net.eval()
        self.net(e_sample, self.current_task)
        threshold = self.threshold_first if self.current_task == 0 else self.threshold
        self.net.decompose(threshold, self.device, is_update_basis=self.args.is_update_basis)
        decompose_time = start_time - time.time()

        # retrain
        start_time = time.time()
        self.net.set_mode('retrain')
        self.train_(train_loader, epochs=self.retrain_epochs, is_retrain=True)
        retrain_time = start_time - time.time()

        print('train_time:', train_time, 'decompose_time:', decompose_time, 'retrain_time:', retrain_time)

    def train_(self, train_loader, epochs, is_retrain=False):
        begin_cat = self.t_c_arr[self.current_task][0]
        end_cat = self.t_c_arr[self.current_task][-1] + 1
        if self.args.is_rotate:
            begin_cat *= 4
            end_cat *= 4
        print('begin_cat', begin_cat, 'end_cat', end_cat)

        self.net.train()

        lf = self.learning_rate
        if self.current_task == 0:
            opt = torch.optim.SGD(self.net.parameters(), lr=lf)
        else:
            opt = torch.optim.SGD(self.net.get_params(), lr=lf)
        scheduler = StepLR(opt, step_size=45, gamma=0.1)
        for epoch in range(int(epochs)):
            for step, data in enumerate(train_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                if self.args.is_rotate:
                    # self-supervised learning based label augmentation
                    inputs = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1)
                    if self.args.dataset == 'seq-cifar100' or self.args.dataset == 'seq-cifar10':
                        inputs = inputs.view(-1, 3, 32, 32)
                    elif self.args.dataset == 'seq-mnist':
                        inputs = inputs.view(-1, 1, 28, 28)
                    elif self.args.dataset == 'seq-tinyimg':
                        inputs = inputs.view(-1, 3, 64, 64)
                    labels = torch.stack([labels * 4 + k for k in range(4)], 1).view(-1)

                outputs = self.net(inputs, self.current_task)
                # print(outputs.shape, begin_cat, end_cat)

                loss_ce = self.loss(-outputs[:, begin_cat:end_cat], labels - begin_cat)

                loss_mu = torch.tensor(0.0).to(self.device)
                if not is_retrain and self.current_task > 0:
                    kk = 0
                    for k, conv in enumerate(self.net.get_convs()):
                        weight = conv.weight
                        weight = weight.view(weight.data.shape[0], -1)

                        basis = conv.basis_
                        basis_t = conv.basis_.transpose(0, 1)

                        weight_proj = torch.mm(basis_t, weight)
                        weight_sub = weight - torch.mm(basis, weight_proj)

                        weight_ = torch.cat([
                            basis_t,
                            weight_sub.transpose(0, 1),
                        ], dim=0)

                        corr = torch.mm(
                            weight_.transpose(0, 1),
                            weight_
                        )
                        loss_mu += (torch.sum(torch.diag(corr)) / torch.sum(corr))

                        kk += 1

                loss = loss_ce + self.lambd * loss_mu

                opt.zero_grad()
                loss.backward()
                opt.step()

            scheduler.step()
            if epoch % self.args.print_freq == 0:
                print('epoch:%d, loss:%.5f, loss_ce:%.5f, loss_mu:%.5f' % (
                    epoch, loss.to('cpu').item(), loss_ce.to('cpu').item(), loss_mu.to('cpu').item()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        self.net.set_mode('predict')
        x = x.to(self.device)
        with torch.no_grad():
            outputs = self.net(x, self.current_task + 1)
            if self.args.is_rotate:
                outputs = outputs[:, ::4] # only compute predictions on original class nodes

        end_cat = self.t_c_arr[self.current_task][-1] + 1
        outputs = -outputs[:, :end_cat]
        return outputs

    def origin_forward(self, x: torch.Tensor) -> torch.Tensor:
        self.net.eval()
        self.net.set_mode('retrain')
        digits = []
        for task_id in range(self.current_task + 1):
            x = x.to(self.device)
            with torch.no_grad():
                begin_cat = self.t_c_arr[task_id][0]
                end_cat = self.t_c_arr[task_id][-1] + 1

                outputs = self.net(x, task_id)
                if self.args.is_rotate:
                    outputs = outputs[:, ::4]  # only compute predictions on original class nodes

                digits.append(-outputs[:, begin_cat:end_cat])
        digits = torch.cat(digits, dim=1)
        return digits

    def end_il(self, dataset):

        for bs in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
            sample = torch.rand([bs, dataset.n_channel, dataset.n_imsize1, dataset.n_imsize2]).to(self.device)
            start_time = time.time()
            self.forward(sample)
            predict_time = time.time() - start_time

            start_time = time.time()
            self.origin_forward(sample)
            origin_predict_time = time.time() - start_time

            print('batch_size:', bs, 'predict_time:', predict_time, 'origin_predict_time:', origin_predict_time)


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


def conv3x3(in_planes, out_planes, stride=1):
    return MyConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                    padding=1)


def conv7x7(in_planes, out_planes, stride=1):
    return MyConv2d(in_planes, out_planes, kernel_size=7, stride=stride,
                    padding=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, nt, stride=1, is_bn_stats=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)

        bn_list = []
        for i in range(nt):
            bn_list.append(nn.BatchNorm2d(planes, track_running_stats=is_bn_stats, affine=True))
        self.bn1s = nn.ModuleList(bn_list)
        self.conv2 = conv3x3(planes, planes)

        bn_list = []
        for i in range(nt):
            bn_list.append(nn.BatchNorm2d(planes, track_running_stats=is_bn_stats, affine=True))
        self.bn2s = nn.ModuleList(bn_list)

        self.dropout = nn.Dropout(p=0.1)

        self.shortcut = nn.Sequential()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride
        if self.stride != 1 or self.in_planes != self.expansion * self.planes:
            bn_list = []
            for i in range(nt):
                bn_list.append(nn.BatchNorm2d(self.expansion * planes, track_running_stats=is_bn_stats, affine=True))
            self.bn3s = nn.ModuleList(bn_list)

            self.conv3 = MyConv2d(in_planes, self.expansion * planes, kernel_size=1,
                                  stride=stride)
        self.count = 0
        self.mode = 'train'

    def forward(self, x, task, isprint=False):
        if self.mode == 'train' or self.mode == 'retrain':
            task_id = task

            out = self.conv1(x, task_id)
            # if isprint: print('conv1', out)
            out = relu(self.bn1s[task_id](out))
            # if isprint: print('bn1s', out)
            out = self.conv2(out, task_id)
            # if isprint: print('conv2', out)
            out = self.bn2s[task_id](out)
            # if isprint: print('bn2s', out)

            if self.stride != 1 or self.in_planes != self.expansion * self.planes:
                sc_out = self.conv3(x, task_id)
                # if isprint: print('sc_out', sc_out)
                out += self.bn3s[task_id](sc_out)
                # if isprint: print('out', out)

            out = relu(out)
        else:
            task_num = task

            out = self.conv1(x, task_num)
            for task_id in range(task_num):
                c_per_task = out.shape[1] // task_num
                out[:, c_per_task * task_id:c_per_task * (task_id + 1), :, :] = self.bn1s[task_id](
                    out[:, c_per_task * task_id:c_per_task * (task_id + 1), :, :])
            out = relu(out)

            out = self.conv2(out, task_num)
            for task_id in range(task_num):
                c_per_task = out.shape[1] // task_num
                out[:, c_per_task * task_id:c_per_task * (task_id + 1), :, :] = self.bn2s[task_id](
                    out[:, c_per_task * task_id:c_per_task * (task_id + 1), :, :])

            if self.stride != 1 or self.in_planes != self.expansion * self.planes:
                sc_out = self.conv3(x, task_num)

                for task_id in range(task_num):
                    c_per_task = out.shape[1] // task_num
                    out[:, c_per_task * task_id:c_per_task * (task_id + 1), :, :] += self.bn3s[task_id](
                        sc_out[:, c_per_task * task_id:c_per_task * (task_id + 1), :, :])

            out = relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, nc, nt, nf, img_size, is_bn_stats=True):
        super(ResNet, self).__init__()
        self.nt, self.nc = nt, nc
        self.is_bn_stats = is_bn_stats

        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1, 1)

        bn_list = []
        for i in range(nt):
            bn_list.append(nn.BatchNorm2d(nf * 1, track_running_stats=is_bn_stats, affine=True))
        self.bn1s = nn.ModuleList(bn_list)

        self.dropout = nn.Dropout(p=0.1)
        self.blocks = []

        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        self.img_size = img_size
        self.linear = nn.Linear(nf * 8 * block.expansion * int(img_size / 16) * int(img_size / 16), nc, bias=False)
        self.c = nn.Parameter(torch.ones(nc), requires_grad=True)

        self.mode = 'train'

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            block_ = block(self.in_planes, planes, self.nt, stride, self.is_bn_stats)
            layers.append(block_)
            self.in_planes = planes * block.expansion
            self.blocks.append(block_)
        return nn.Sequential(*layers)

    def forward(self, x, task):

        if self.mode == 'train' or self.mode == 'retrain':
            task_id = task
            bsz = x.size(0)
            out = x.view(bsz, 3, self.img_size, self.img_size)

            out = self.conv1(out, task_id)
            out = relu(self.bn1s[task_id](out))

            for block in self.blocks:
                out = block(out, task_id)

            out = avg_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            y = self.linear(out)
            y = (y - self.c) ** 2
        else:
            task_num = task

            bsz = x.size(0)
            out = x.view(bsz, 3, self.img_size, self.img_size)
            out = out.repeat(1, task_num, 1, 1)

            out = self.conv1(out, task_num)
            for task_id in range(task_num):
                c_per_task = out.shape[1] // task_num
                out[:, c_per_task * task_id:c_per_task * (task_id + 1), :, :] \
                    = self.bn1s[task_id](out[:, c_per_task * task_id:c_per_task * (task_id + 1), :, :])
            out = relu(out)

            for block in self.blocks:
                out = block(out, task_num)

            y = torch.zeros([bsz, self.nc]).to(out)
            out = avg_pool2d(out, 2)
            for task_id in range(task_num):
                c_per_task = out.shape[1] // task_num
                out_temp = out[:, c_per_task * task_id:c_per_task * (task_id + 1), :, :]

                cpt = self.nc // self.nt
                out_temp = out_temp.view(out_temp.size(0), -1)
                y[:, cpt * task_id:cpt * (task_id + 1)] = self.linear(out_temp)[:, cpt * task_id:cpt * (task_id + 1)]

            y = (y - self.c) ** 2

        return y

    def get_params(self) -> torch.Tensor:
        params_arr = []
        conv_list = self.get_convs()
        for conv in conv_list:
            params_arr.append(
                {"params": conv.parameters()}
            )
        # params_arr.append({"params": self.linear.parameters()})
        return params_arr

    def decompose(self, threshold, device, is_update_basis=True):
        total_basis_size, total_weight_size = 0.0, 0.0

        conv_list = self.get_convs()
        for conv in conv_list:
            conv.decompose(threshold, device, is_update_basis)

            total_basis_size += conv.basis_.shape[0] * conv.basis_.shape[1]
            for weight_ in conv.weight_:
                total_weight_size += weight_.shape[0] * weight_.shape[1]

        print('total_basis_size', total_basis_size, 'total_weight_size', total_weight_size)

    def get_convs(self):
        conv_list = [self.conv1]
        for block in self.blocks:
            conv_list.append(block.conv1)
            conv_list.append(block.conv2)
            if hasattr(block, 'conv3'):
                conv_list.append(block.conv3)

        return conv_list

    def set_mode(self, mode):
        self.mode = mode
        conv_list = self.get_convs()

        for idx, conv in enumerate(conv_list):
            conv.set_mode(mode)

        for block in self.blocks:
            block.mode = mode


def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1):
    return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))


def ResNet18(nc, nt, nf=64, img_size=32, is_bn_stats=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nc, nt, nf, img_size, is_bn_stats)


def ResNet34(nc, nt, nf=64, img_size=32, is_bn_stats=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], nc, nt, nf, img_size, is_bn_stats)


def get_model(model):
    return deepcopy(model.state_dict())


def set_model_(model, state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return


class MyConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MyConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        kernel_size_ = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = Parameter(torch.empty(
            (out_channels, in_channels // groups, *kernel_size_), **factory_kwargs))

        self.weight_ = []
        self.basis_ = None

        self.A_conv = None
        self.B_conv = None

        self.register_parameter('bias', None)
        self.reset_parameters()

        self.z = None

        # train or predict
        self.mode = 'train'

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def set_mode(self, mode):
        self.mode = mode

    def forward(self, input: Tensor, task) -> Tensor:
        if self.mode == 'train':
            weight = self.weight

            z = self._conv_forward(input, weight, self.bias)

            self.z = z.to('cpu')  # record input message
        elif self.mode == 'retrain':  # task is the task id
            weight_ = self.weight_[task]
            valid_dim = weight_.shape[0]
            basis_ = self.basis_[:, :valid_dim]

            shape = self.weight.shape
            weight = torch.mm(basis_, weight_).view(shape)

            z = self._conv_forward(input, weight, self.bias)
        else:  # task is the task number
            bsz, _, _, _ = input.shape

            z = F.conv2d(input, self.A, None, self.stride, self.padding, self.dilation, task)

            z = z.view(bsz * task, -1, z.shape[2], z.shape[3])
            z = F.conv2d(z, self.B)
            z = z.view(bsz, -1, z.shape[2], z.shape[3])

        return z

    def decompose(self, threshold, device, is_update_basis=True):
        act = self.z
        shape = act.shape
        mat = torch.transpose(act, 0, 1)
        mat = mat.contiguous().view(shape[1], -1)
        activation = mat.detach().cpu().numpy()

        if self.basis_ is None:
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)

            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold)  # +1
            if r == 0:
                r = 1
            basis_ = torch.Tensor(U[:, 0:r]).to(device)
            self.basis_ = nn.Parameter(basis_, requires_grad=False)
        elif is_update_basis:
            basis = self.basis_.detach().cpu().float().numpy()

            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1 ** 2).sum()

            act_hat = activation - np.dot(np.dot(
                basis,
                basis.transpose()
            ), activation)
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)

            sval_hat = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            accumulated_sval = (sval_total - sval_hat) / sval_total

            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break

            if r == 0:
                # print('Skip Updating Basis')
                pass
            else:
                # update basis
                Ui = np.hstack((basis, U[:, 0:r]))
                if Ui.shape[1] > Ui.shape[0]:
                    basis_ = torch.Tensor(Ui[:, 0:Ui.shape[0]]).to(device)
                else:
                    basis_ = torch.Tensor(Ui).to(device)

                self.basis_ = nn.Parameter(basis_, requires_grad=False)

        sz = self.weight.data.size(0)
        weight = nn.Parameter(torch.mm(
            self.basis_.transpose(0, 1),
            self.weight.data.view(sz, -1)
        ), requires_grad=True)

        self.weight_.append(nn.Parameter(weight, requires_grad=True))

        print('basis_size:', self.basis_.shape, 'weight_size', weight.shape)

        # ----- convert weight and basis into convs -----
        c_out, c_in, k, _ = self.weight.shape
        _, m = self.basis_.shape

        weight_arr = []
        for weight_ in self.weight_:
            weight_ = weight_.view(-1, c_in, k, k)
            m_t = weight_.shape[0]

            weight_t = torch.zeros((m, c_in, k, k)).to(self.weight)
            weight_t[:m_t, ...] = weight_
            weight_arr.append(weight_t)
        self.A = torch.cat(weight_arr, dim=0)

        self.B = self.basis_.view(c_out, m, 1, 1)


class MyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight_ = None
        self.basis_ = None
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.z = None

        # train or predict
        self.mode = 'train'

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:

        if self.mode == 'train':
            weight = self.weight

            z = F.linear(input, weight, self.bias)

            self.z = z.to('cpu')  # record input message
        elif self.mode == 'retrain':  # task is the task id
            weight_ = self.weight_[task]
            valid_dim = weight_.shape[0]
            basis_ = self.basis_[:, :valid_dim]

            shape = self.weight.shape
            weight = torch.mm(basis_, weight_).view(shape)

            z = F.linear(input, weight, self.bias)
        else:  # task is the task number
            bsz, _, _, _ = input.shape

            z = F.linear(input, self.A)
            z = F.linear(z, self.B)

        return z

    def decompose(self, threshold, device, is_update_basis=True):
        act = self.z
        shape = act.shape
        mat = torch.transpose(act, 0, 1)
        mat = mat.contiguous().view(shape[1], -1)
        activation = mat.detach().cpu().numpy()

        if self.basis_ is None:
            U, S, Vh = np.linalg.svd(activation, full_matrices=False)

            sval_total = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            r = np.sum(np.cumsum(sval_ratio) < threshold)  # +1
            basis_ = torch.Tensor(U[:, 0:r]).to(device)
            self.basis_ = nn.Parameter(basis_, requires_grad=False)
        elif is_update_basis:
            basis = self.basis_.detach().cpu().float().numpy()

            U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
            sval_total = (S1 ** 2).sum()

            act_hat = activation - np.dot(np.dot(
                basis,
                basis.transpose()
            ), activation)
            U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)

            sval_hat = (S ** 2).sum()
            sval_ratio = (S ** 2) / sval_total
            accumulated_sval = (sval_total - sval_hat) / sval_total

            r = 0
            for ii in range(sval_ratio.shape[0]):
                if accumulated_sval < threshold:
                    accumulated_sval += sval_ratio[ii]
                    r += 1
                else:
                    break

            if r == 0:
                # print('Skip Updating Basis')
                pass
            else:
                # update basis
                Ui = np.hstack((basis, U[:, 0:r]))
                if Ui.shape[1] > Ui.shape[0]:
                    basis_ = torch.Tensor(Ui[:, 0:Ui.shape[0]]).to(device)
                else:
                    basis_ = torch.Tensor(Ui).to(device)

                self.basis_ = nn.Parameter(basis_, requires_grad=False)

        sz = self.weight.data.size(0)
        weight = nn.Parameter(torch.mm(
            self.basis_.transpose(0, 1),
            self.weight.data.view(sz, -1)
        ), requires_grad=True)

        self.weight_.append(nn.Parameter(weight, requires_grad=True))

        print('basis_size:', self.basis_.shape, 'weight_size', weight.shape)

        # ----- convert weight and basis into fcs -----
        d_out, d_in = self.weight.shape
        _, m = self.basis_.shape

        weight_arr = []
        for weight_ in self.weight_:
            m_t = weight_.shape[0]

            weight_t = torch.zeros((m, d_in)).to(self.weight)
            weight_t[:m_t, ...] = weight_
            weight_arr.append(weight_t)
        self.A = torch.cat(weight_arr, dim=0)

        self.B = self.basis_

class LeNet(nn.Module):
    def __init__(self, nt, cpt, nf=10):
        super(LeNet, self).__init__()
        self.nf = nf

        self.act = OrderedDict()
        self.map = []
        self.ksize = []
        self.in_channel = []

        self.map.append(32)
        self.conv1 = MyConv2d(3, self.nf * 2, kernel_size=5, padding=2, bias=False)

        s = compute_conv_output_size(32, 5, 1, 2)
        s = compute_conv_output_size(s, 3, 2, 1)
        self.ksize.append(5)
        self.in_channel.append(3)
        self.map.append(s)
        self.conv2 = MyConv2d(self.nf * 2, self.nf * 5, kernel_size=5, padding=2, bias=False)

        s = compute_conv_output_size(s, 5, 1, 2)
        s = compute_conv_output_size(s, 3, 2, 1)
        self.ksize.append(5)
        self.in_channel.append(self.nf * 2)
        self.smid = s
        self.map.append(self.nf * 5 * self.smid * self.smid)
        self.maxpool = torch.nn.MaxPool2d(3, 2, padding=1)
        self.relu = torch.nn.ReLU()
        self.drop1 = torch.nn.Dropout(0)
        self.drop2 = torch.nn.Dropout(0)
        self.lrn = torch.nn.LocalResponseNorm(4, 0.001 / 9.0, 0.75, 1)

        self.fc1 = MyLinear(self.nf * 5 * self.smid * self.smid, self.nf * 80, bias=False)
        self.fc2 = MyLinear(self.nf * 80, self.nf * 50, bias=False)
        self.map.extend([self.nf * 80])

        self.fc3 = torch.nn.Linear(self.nf * 50, cpt * nt, bias=True)

        self.mode = 'train'

    def forward(self, x, task_id):
        bsz = deepcopy(x.size(0))
        x = self.conv1(x)
        x = self.maxpool(self.lrn(self.relu(x)))

        x = self.conv2(x)
        x = self.maxpool(self.lrn(self.relu(x)))

        x = x.reshape(bsz, -1)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        y = self.fc3(x)

        y = (y - self.c) ** 2

        return y

    def get_params(self) -> torch.Tensor:
        params_arr = []
        conv_list = self.get_convs()
        for conv in conv_list:
            params_arr.append(
                {"params": conv.parameters()}
            )
        return params_arr

    def get_convs(self):
        return [self.conv1, self.conv2, self.fc1, self.fc2]

    def set_mode(self, mode):
        self.mode = mode
        conv_list = self.get_convs()

        for idx, conv in enumerate(conv_list):
            conv.set_mode(mode)

        for block in self.blocks:
            block.mode = mode