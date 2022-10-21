from engine_phases import train_distill, eval_probing, train_task
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
    parser.add_argument('--dataset_name', type=str, default="ogbg-moltox21",
                        help='dataset name (default: ogbg-molhiv/moltox21/molpcba)')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--track_all', action='store_true',
                        help='whether track topsim and msg entropy') 
    # ==== Model settings ======
    #===========================
    parser.add_argument('--backbone_type', type=str, default='gcn',
                        help='backbone type, can be gcn, gin, gcn_virtual, gin_virtual')
    parser.add_argument('--bottle_type', type=str, default='upsample',
                        help='bottleneck type, can be pool, upsample, ...')
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
    parser.add_argument('--epochs_ft', type=int, default=20,
                        help='student training on real label')
    parser.add_argument('--epochs_dis', type=int, default=20,
                        help='distillation')
    parser.add_argument('--generations', type=int, default=5,
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
              help='how the teacher generate the samples, ce_argmax, ce_sample, noisy_ce_sample, mse, kld')
    parser.add_argument('--dis_optim_type', type=str, default='adam',
              help='optimizer used in distillation, sgd or adam')
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
    parser.add_argument('--ckp_name',default='_best', type=str)
    return parser
 
def main(args):
    # Only analyze the detailed behaviors during distillation
    # ========== Generate seed ==========
    if args.seed==0:
        args.seed = np.random.randint(1,10086)
    rnd_seed(args.seed)
    
    # ========== Prepare save folder and wandb ==========
    run_name = wandb_init(proj_name=args.proj_name, run_name=args.run_name, config_args=args)
    model_name = args.backbone_type+'_'+args.bottle_type
    args.save_path = os.path.join('results',model_name,args.dataset_name)
            # -------- save results in this folder
    # ========== Prepare the loader, model and optimizer
    loaders = build_dataset(args)    
    student = get_init_net(args)
    teacher = get_init_net(args)
        # ------ Here we load the checkpoint when analyzing each stage
    ckp_name = args.ckp_name
    ckp_load_path = os.path.join(args.save_path,model_name+'_'+args.dataset_name+ckp_name+'.pth')
    teacher.load_state_dict(torch.load(ckp_load_path),strict=True)
    
    if args.dis_optim_type == 'sgd':
        optimizer_dis = optim.SGD(student.parameters(), momentum=0.9, weight_decay=args.dis_wd, lr=args.dis_lr)
    elif args.dis_optim_type == 'adam':
        optimizer_dis = optim.Adam(student.parameters(), lr=args.dis_lr)
    # ===== Distill
    eval_probing(args, teacher, loaders, title='Teach_', no_train=True)
    eval_probing(args, student, loaders, title='Stud_prob_', no_train=True)
    for epoch in range(args.epochs_dis):
        print(epoch,end='-')
        train_distill(args, student, teacher, loaders['train'], optimizer_dis)
        eval_probing(args, student, loaders, title='Stud_prob_', no_train=True)
    wandb.finish()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    main(args)


