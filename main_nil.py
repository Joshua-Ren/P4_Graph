# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:34:13 2022

@author: YIREN
"""
from engine_phases import train_distill, eval_probing, train_task, eval_all
from utils.datasets import build_dataset
from utils.general import wandb_init, get_init_net, rnd_seed, AverageMeter, early_stop_meets
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
    parser.add_argument('--WD_ID',default='joshuaren', type=str,
                        help='W&D ID, joshuaren or joshua_shawn')
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
                        help='bottleneck type, can be pool, upsample, updown, lstm, ...')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')    
        # ---- SEM settings ----
    parser.add_argument('--L', type=int, default=15,
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
    parser.add_argument('--teach_last_best', type=str, default='best',
                        help='use the best or last epoch teacher in distillation')
    parser.add_argument('--epochs_lp', type=int, default=1,
                        help='for lp probing epochs')  
    parser.add_argument('--epochs_ssl', type=int, default=0,
                        help='byol between two models')
    parser.add_argument('--epochs_ft', type=int, default=5,
                        help='student training on real label, >500 is early stopping')
    parser.add_argument('--es_epochs', type=int, default=3,
                        help='consecutive how many epochs non-increase')
    parser.add_argument('--epochs_dis', type=int, default=2,
                        help='distillation')
    parser.add_argument('--generations', type=int, default=20,
                        help='number of generations')
  
        # ===== Finetune or evaluation settings ======
    parser.add_argument('--ft_lr', type=float, default=1e-3,
                        help='learning rate for student on task')
    parser.add_argument('--lp_lr', type=float, default=1e-3,
                        help='learning rate for student when LP-eval')
        # ===== Distillation settings ======
    parser.add_argument('--dis_lr', type=float, default=1e-3,
                        help='learning rate for student')    
    parser.add_argument('--scheduler', type=eval, default=True,
                        help='Whether to use cosine scheduler')    
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
    return parser

def main(args):
    # Model and optimizers are build in
    # In each generation:
    #   Step0: prepare everything
    #   Step1: distillation, skip if first gen
    #   [Step2: student SSL like SimCLR]
    #   Step2: student ft on task
    #   Step3: student becomes the teacher
    '''
    ft_losses = AverageMeter()
    ft_msg_dists = AverageMeter()
    ft_msg_topsim = AverageMeter()
    ft_msg_entropy = AverageMeter()
    ft_train_roc = AverageMeter()
    dis_losses = AverageMeter()
    dis_msg_dists = AverageMeter()
    dis_msg_topsim = AverageMeter()
    dis_msg_entropy = AverageMeter()
    '''
    # ========== Generate seed ==========
    if args.seed==0:
        args.seed = np.random.randint(1,10086)
    rnd_seed(args.seed)
    
    # ========== Prepare save folder and wandb ==========
    run_name = wandb_init(proj_name=args.proj_name, run_name=args.run_name, config_args=args)
    model_name = args.backbone_type+'_'+args.bottle_type
    args.save_path = os.path.join('results',model_name,args.dataset_name)    
    # ========== Prepare the loader and optimizer
    loaders = build_dataset(args)    
    #results = {'End_gen_train_roc':[],'End_gen_valid_roc':[],'End_gen_test_roc':[],
    #           'best_val_epoch':0,'best_val_roc':0,'best_test_roc':0}
    
    for gen in range(args.generations):
        # =========== Step0: new agent
        student = get_init_net(args)
        if args.dis_optim_type=='adam':
            optimizer_dis = optim.Adam(student.parameters(), lr=args.dis_lr)
        elif args.dis_optim_type=='sgd':
            optimizer_dis = optim.SGD(student.parameters(), momentum=0.9, lr=args.dis_lr)
        optimizer_ft = optim.Adam(student.parameters(), lr=args.ft_lr)      
        #optimizer_ft = optim.Adam(student.parameters(), lr=args.ft_lr)
        if args.scheduler:
            if args.epochs_ft>1000:
                tmax = 100
            else:
                tmax = args.epochs_ft
            scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft,T_max=tmax,eta_min=1e-6)
        else:
            scheduler_ft = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft,T_max=100,eta_min=args.ft_lr)
        # =========== Step1: distillation, skip in first gen
        if gen>0:
            for epoch in range(args.epochs_dis):
                print(epoch,end='-')
                train_distill(args, student, teacher, loaders['train'], optimizer_dis)
                #eval_probing(args, student, loaders, title='Stud_prob_', no_train=True)
        
        # =========== Step2: solve task, track best valid acc
        if args.track_all:
            student0 = copy.deepcopy(student)       # Use this to track the change of message
        else:
            student0 = None
        best_vacc, best_vacc_ep, best_testacc, vacc_list = 0, 0, 0, []
        for epoch in range(args.epochs_ft):
            args.ft_tau = 4/(epoch+1)
            train_task(args, student, loaders['train'], optimizer_ft, scheduler_ft, student0)
            train_roc, valid_roc, test_roc = eval_all(args, student, loaders, title='ft_', no_train=True)
            vacc_list.append(valid_roc)
            if valid_roc > best_vacc:
                best_vacc = valid_roc
                best_testacc = test_roc
                best_vacc_ep = epoch
                if args.teach_last_best=='best':
                    teacher = copy.deepcopy(student)
            wandb.log({'best_val_epoch':best_vacc_ep})
            # ------- Early stop the FT if 3 non-increasing epochs
            if args.epochs_ft>500 and early_stop_meets(vacc_list, best_vacc, how_many=args.es_epochs):
                break
            if epoch>100:
                break
        if args.teach_last_best=='last':
            teacher = copy.deepcopy(student)     
        wandb.log({'End_gen_valid_roc':valid_roc})
        wandb.log({'End_gen_test_roc':test_roc})
        wandb.log({'Best_gen_valid_roc':best_vacc})
        wandb.log({'Best_gen_test_roc':best_testacc})
        del student0
    wandb.finish()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.epochs_ft = 1000
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    main(args)





  
