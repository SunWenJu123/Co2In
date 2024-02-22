import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from argparse import ArgumentParser

from utils.args import get_args
from utils.training import train_il
from utils.conf import set_random_seed
import torch


def main():
    args = get_args()
    args.seed = None
    args.validation = False
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        set_random_seed(args.seed)

    # BMKP
    args.model = 'bmkp'
    args.img_dir = 'img/bmkp/'
    if not os.path.exists(args.img_dir):
        os.makedirs(args.img_dir)

    args.dataset = 'seq-cifar10'
    args.print_freq = 10
    args.n_epochs = 50
    args.retrain_epochs = 50
    args.net = 'resnet18'

    args.lr = 5e-2
    args.batch_size = 128
    args.threshold_first = 0.96
    args.threshold = 0.96
    args.nf = 20
    args.lambd = 10
    args.skip_layers = list(range(16))
    args.is_bn_stats = False
    args.is_update_basis = True
    args.example_num = 500


    args.dataset = 'seq-cifar100'
    args.print_freq = 10
    args.n_epochs = 100
    args.retrain_epochs = 100
    args.net = 'resnet18'

    args.lr = 5e-2
    args.batch_size = 128
    args.threshold = 0.96
    args.threshold_first = 0.95
    args.nf = 20
    args.lambd = 1
    args.is_bn_stats = False
    args.is_update_basis = True
    args.example_num = 500


    args.dataset = 'seq-tinyimg'
    args.print_freq = 10
    args.n_epochs = 100
    args.retrain_epochs = 100
    args.net = 'resnet18'

    args.lr = 3e-2
    args.batch_size = 64
    args.threshold = 0.957
    args.threshold_first = 0.97
    args.nf = 20
    args.lambd = 10
    args.skip_layers = list(range(16))
    args.is_bn_stats = True
    args.is_update_basis = True
    args.example_num = 500

    # Co2In
    args.model = 'bmkpv2'
    args.img_dir = 'img/bmkpv2/'
    if not os.path.exists(args.img_dir):
        os.makedirs(args.img_dir)

    args.dataset = 'seq-cifar10'
    args.print_freq = 10
    args.n_epochs = 50
    args.retrain_epochs = 50

    args.lr = 0.1  # ok
    args.batch_size = 128
    args.threshold_first = 0.98
    args.threshold = 0.97
    args.nf = 20
    args.lambd = 0
    args.is_bn_stats = True
    args.is_update_basis = True
    args.is_rotate = True
    args.example_num = 500
    args.c_reg_lambd = 0


    args.dataset = 'seq-cifar100'
    args.print_freq = 10
    args.n_epochs = 100
    args.retrain_epochs = 100
    args.is_rotate = False

    args.lr = 2e-2
    args.batch_size = 16
    args.threshold = 0.97
    args.threshold_first = 0.98
    args.is_update_basis = True
    args.nf = 20
    args.lambd = 1
    args.is_bn_stats = True
    args.example_num = 500
    args.c_reg_lambd = 0


    args.dataset = 'seq-tinyimg'
    args.print_freq = 10
    args.n_epochs = 100
    args.retrain_epochs = 100
    args.net = 'resnet18'

    args.lr = 3e-2
    args.batch_size = 128
    args.threshold = 0.98
    args.threshold_first = 0.98
    args.nf = 20
    args.lambd = 10
    args.is_bn_stats = True
    args.is_update_basis = True
    args.is_rotate = False
    args.example_num = 500
    args.c_reg_lambd = 0

    for conf in [1, 2, 3, 4, 5]:
        print("")
        print("=================================================================")
        print("==========================", "seq-cifar10", ":", conf, "==========================")
        print("=================================================================")
        print("")
        train_il(args)


if __name__ == '__main__':
    main()
