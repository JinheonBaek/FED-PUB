import os
from parser import Parser
from datetime import datetime

from misc.utils import *
from modules.multiprocs import ParentProcess

def main(args):

    args = set_config(args)

    if args.model == 'fedavg':    
        from models.fedavg.server import Server
        from models.fedavg.client import Client
    elif args.model == 'fedpub':    
        from models.fedpub.server import Server
        from models.fedpub.client import Client
    else:
        print('incorrect model was given: {}'.format(args.model))
        os._exit(0)

    pp = ParentProcess(args, Server, Client)
    pp.start()

def set_config(args):

    args.base_lr = 1e-3
    args.min_lr = 1e-3
    args.momentum_opt = 0.9
    args.weight_decay = 1e-6
    args.warmup_epochs = 10
    args.base_momentum = 0.99
    args.final_momentum = 1.0

    if args.dataset == 'Cora':
        args.n_feat = 1433
        args.n_clss = 7
        args.n_clients = 10 if args.n_clients == None else args.n_clients
        args.base_lr = 0.01 if args.lr == None else args.lr

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial = f'{args.dataset}_{args.mode}/clients_{args.n_clients}/{now}_{args.model}'

    args.data_path = f'{args.base_path}/datasets' 
    args.checkpt_path = f'{args.base_path}/checkpoints/{trial}'
    args.log_path = f'{args.base_path}/logs/{trial}'

    if args.debug == True:
        args.checkpt_path = f'{args.base_path}/debug/checkpoints/{trial}'
        args.log_path = f'{args.base_path}/debug/logs/{trial}'

    return args

if __name__ == '__main__':
    main(Parser().parse())










