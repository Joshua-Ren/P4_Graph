from engine_phases import train_distill, eval_probing, train_task, eval_all
from utils.datasets import build_dataset
from utils.general import wandb_init, get_init_net, rnd_seed, AverageMeter
from utils.nil_related import *
import torch.optim as optim
import torch
import argparse
import numpy as np
import random
import os
import wandb

def get_args_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset_name', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv/moltox21/molpcba)')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')

    # ==== Model settings ======
    #===========================
    parser.add_argument('--backbone_type', type=str, default='gcn',
                        help='backbone type, can be gcn, gin, gcn_virtual, gin_virtual')
    parser.add_argument('--bottle_type', type=str, default='gumbel',
                        help='bottleneck type, can be pool, upsample, gumbel ...')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')    
        # ---- SEM settings ----
    parser.add_argument('--L', type=int, default=200,
                        help='No. word in SEM')
    parser.add_argument('--V', type=int, default=20,
                        help='word size in SEM')
    parser.add_argument('--pool_tau', type=float, default=1.,
                        help='temperature in SEM')
    parser.add_argument('--dis_sem_tau', type=float, default=1.,
                        help='temperature in SEM')
    parser.add_argument('--ft_tau', type=float, default=1.,
                        help='temperature in SEM')
    
    # ===== NIL settings ======
    # =========================
    parser.add_argument('--epochs_lp', type=int, default=1,
                        help='for lp probing epochs')  
    parser.add_argument('--epochs_ssl', type=int, default=0,
                        help='byol between two models')
    parser.add_argument('--epochs_ft', type=int, default=0,
                        help='student training on real label')
    parser.add_argument('--epochs_dis', type=int, default=100,
                        help='distillation')
    parser.add_argument('--generations', type=int, default=2,
                        help='number of generations')
    
        # ===== Finetune or evaluation settings ======
    parser.add_argument('--ft_lr', type=float, default=1e-3,
                        help='learning rate for student on task')
    parser.add_argument('--lp_lr', type=float, default=1e-3,
                        help='learning rate for student when LP-eval')
        # ===== Distillation settings ======
    parser.add_argument('--dis_lr', type=float, default=1e-3,
                        help='learning rate for student')    
    parser.add_argument('--dis_loss', type=str, default='ce_argmax',
              help='how the teacher generate the samples, ce_argmax, ce_sample, mse, kld')
    parser.add_argument('--dis_smp_tau', type=float, default=1.,
              help='temperature used when teacher generating sample, 0 is argmax')
    
        # ===== SSL settings ======
            # ---- Common
    parser.add_argument('--inter_alpha', type=float, default=0,
                        help='balance between task loss and inter-loss, 0 is all inter')
    parser.add_argument('--ssl_lr', type=float, default=1e-3,
                        help='learning rate for student')                      
            # ---- BYOL
    parser.add_argument('--byol_loss', type=str, default='mse',
                        help='loss type used in byol, mse or cosine')
    parser.add_argument('--byol_eta', type=float, default=0.99,
                        help='eta in EMA')  
    
    # ===== Wandb and saving results ====
    parser.add_argument('--run_name',default='test',type=str)
    parser.add_argument('--proj_name',default='P4_phase_observe', type=str)
    parser.add_argument('--save_dir', default=None,
                        help='path of the pretrained checkpoint')    
    return parser


def main(args, n_epoch=1):
    # ========== Generate seed ==========
    if args.seed==0:
        args.seed = np.random.randint(1,10086)
    rnd_seed(args.seed)
    
    # ========== Prepare save folder and wandb ==========
    run_name = wandb_init(proj_name=args.proj_name, run_name=args.run_name, config_args=args)
    model_name = args.backbone_type+'_'+args.bottle_type
    args.save_path = os.path.join('results',model_name,args.dataset_name)
            # -------- save results in this folder
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)    
    
    # ========== Prepare the loader, model and optimizer
    loaders = build_dataset(args)
    model = get_init_net(args)
    model0 = copy.deepcopy(model)       # Use this to track the change of message
    optimizer_ft = optim.Adam(model.parameters(), lr=args.ft_lr)
    scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft,T_max=n_epoch,eta_min=1e-6)
    best_vacc = 0
    # ========== Train the network and save PT checkpoint
    for epoch in range(n_epoch):
        train_task(args, model, loaders['train'], optimizer_ft, scheduler_ft, model0)
        train_roc, valid_roc, test_roc = eval_all(args, model, loaders, title='Stud_', no_train=True)
        if valid_roc > best_vacc:
            best_vacc = valid_roc
            #ckp_save_path = os.path.join(args.save_path,model_name+'_'+args.dataset_name+'_best.pth')
            #torch.save(model.state_dict(),ckp_save_path)        
    #ckp_save_path = os.path.join(args.save_path,model_name+'_'+args.dataset_name+'_last.pth')
    #torch.save(model.state_dict(),ckp_save_path)
    wandb.finish()
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    main(args, n_epoch=100)

