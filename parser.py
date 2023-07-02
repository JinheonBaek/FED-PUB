import argparse
from misc.utils import *

class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()
       
    def set_arguments(self):
        self.parser.add_argument('--gpu', type=str, default='0')
        self.parser.add_argument('--seed', type=int, default=1234)

        self.parser.add_argument('--model', type=str, default=None)
        self.parser.add_argument('--dataset', type=str, default=None)
        self.parser.add_argument('--mode', type=str, default=None, choices=['disjoint', 'overlapping'])
        self.parser.add_argument('--base-path', type=str, default='../')

        self.parser.add_argument('--n-workers', type=int, default=None)
        self.parser.add_argument('--n-clients', type=int, default=None)
        self.parser.add_argument('--n-rnds', type=int, default=None)
        self.parser.add_argument('--n-eps', type=int, default=None)
        self.parser.add_argument('--frac', type=float, default=None)
        self.parser.add_argument('--n-dims', type=int, default=128)
        self.parser.add_argument('--lr', type=float, default=None)

        self.parser.add_argument('--laye-mask-one', action='store_true')
        self.parser.add_argument('--clsf-mask-one', action='store_true')

        self.parser.add_argument('--agg-norm', type=str, default='exp', choices=['cosine', 'exp'])
        self.parser.add_argument('--norm-scale', type=float, default=10)
        self.parser.add_argument('--n-proxy', type=int, default=5)

        self.parser.add_argument('--l1', type=float, default=1e-3)
        self.parser.add_argument('--loc-l2', type=float, default=1e-3)

        self.parser.add_argument('--debug', action='store_true')

    def parse(self):
        args, unparsed  = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args
