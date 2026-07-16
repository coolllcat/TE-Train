""" Config class for search/augment """
import argparse
import os
from functools import partial
import torch

def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text


class AugmentConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Augment config")
        parser.add_argument('--name', default='All_decode')
        parser.add_argument('--epochs', type=int, default=150, help='# of training epochs')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size')
        parser.add_argument('--lr', type=float, default=0.0001, help='lr for weights')

        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma')
        parser.add_argument('--workers', type=int, default=2, help='# of workers')

        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
        parser.add_argument('--grad_clip', type=float, default=5.,help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.path = os.path.join('augments', self.name)
        self.gpus = parse_gpus(self.gpus)

        if not os.path.exists(self.path):
            os.makedirs(self.path)
