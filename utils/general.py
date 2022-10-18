import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
import numpy as np
import torch.backends.cudnn as cudnn
sys.path.append("..")
from models.gnn import *
# ============== Wandb =======================
def wandb_init(proj_name='test', run_name=None, config_args=None):
    wandb.init(
        project=proj_name,
        config={}, entity="joshuaren",reinit=True)
    if config_args is not None:
        wandb.config.update(config_args)
    if run_name is not None:
        wandb.run.name=run_name
        return run_name
    else:
        return wandb.run.name

def get_init_net(args, backbone_type=None, bottle_type=None):
    if backbone_type is None:
        backbone_type = args.backbone_type
    else:
        backbone_type = backbone_type
    if bottle_type is None:
        bot_type = args.bottle_type
    else:
        bot_type = bottle_type
    # ===== backbone type
    if backbone_type=='gcn':
        V_node = False
        G_type = 'gcn'
    elif backbone_type=='gin':
        V_node = False
        G_type = 'gin'
    elif backbone_type=='gcn_virtual':
        V_node = True
        G_type = 'gcn'
    elif backbone_type=='gin_virtual':
        V_node = True
        G_type = 'gin'

    # ===== bottleneck type
    if bot_type == 'pool':
        model = GNN_SEM_POOL(gnn_type=G_type,num_tasks=args.num_tasks, 
                   num_layer=args.num_layer,emb_dim=args.emb_dim,
                   drop_ratio=args.drop_ratio,virtual_node=V_node,
                   L=args.L, V=args.V, tau=args.pool_tau).to(args.device)
    elif bot_type == 'upsample':
        model = GNN_SEM_UPSAMPLE(gnn_type=G_type,num_tasks=args.num_tasks, 
                   num_layer=args.num_layer,emb_dim=args.emb_dim,
                   drop_ratio=args.drop_ratio,virtual_node=V_node,
                   L=args.L, V=args.V).to(args.device)        
    elif bot_type == 'updown':
        model = GNN_SEM_UPDOWN(gnn_type=G_type,num_tasks=args.num_tasks, 
                   num_layer=args.num_layer,emb_dim=args.emb_dim,
                   drop_ratio=args.drop_ratio,virtual_node=V_node,
                   L=args.L, V=args.V).to(args.device)  
    return model

# ============== General functions =======================
def rnd_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True

def save_checkpoint(args, model, file_name='test'):
    pass


def load_checkpoint(args, model, ckp_path, which_part='all'):
    '''
        Use this to load params of specific part (Alice, Bob or all),
        from ckp to model.
    '''
    pass

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
