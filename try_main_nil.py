# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 20:52:55 2022

@author: YIREN
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 16:34:13 2022

@author: YIREN
"""
from engine_phases import train_distill, train_task, evaluate
from utils.datasets import build_dataset
from utils.general import update_args, wandb_init, get_init_net, rnd_seed, AverageMeter, early_stop_meets
from utils.nil_related import *
import torch.optim as optim
import torch
import argparse
import numpy as np
import random
import os
import wandb
import toml

def get_args_parser():
    # Training settings
    # ======= Usually default settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--config_file', type=str, default=None,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--WD_ID',default='joshuaren', type=str,
                        help='W&D ID, joshuaren or joshua_shawn')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset_name', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv/moltox21/molpcba)')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--bottle_type', type=str, default='sem',
                        help='bottleneck type, can be std or sem')
    # ==== Model Structure ======
        # ----- Backbone
    parser.add_argument('--backbone_type', type=str, default='gcn',
                        help='backbone type, can be gcn, gin, gcn_virtual, gin_virtual')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')  
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
        # ---- SEM
    parser.add_argument('--L', type=int, default=15,
                        help='No. word in SEM')
    parser.add_argument('--V', type=int, default=20,
                        help='word size in SEM')
                        
        # ---- Head-type
    parser.add_argument('--head_type', type=str, default='linear',
                        help='Head type in interaction, linear or mlp')    
    
    # ==== NIL related ======    
    parser.add_argument('--generations', type=int, default=10,
                        help='number of generations')
        # ---- Init student
    parser.add_argument('--init_strategy', type=str, default='nil',
                        help='How to generate new student, nil or mile')
    parser.add_argument('--init_part', type=str, default=None,
                        help='Which part of the backbone to re-init')
        # ---- Distillation
    parser.add_argument('--dis_tau', type=float, default=1.,
                        help='temperature used during distillation, same on teacher and student')
    parser.add_argument('--dis_steps', type=int, default=5000,
                        help='distillation batches, epoch should be int(step/N_batches)')
    parser.add_argument('--dis_lr', type=float, default=1e-3,
                        help='learning rate for student')   
    parser.add_argument('--dis_optim', type=str, default='adam',
              help='optimizer used in distillation, sgd, adam or adamW')
    parser.add_argument('--dis_loss', type=str, default='ce_argmax',
              help='how the teacher generate the samples, ce_argmax, ce_sample, noisy_ce_sample, mse')
    parser.add_argument('--distill_data', type=str, default=None,
                        help='dataset name (default: ogbg-molhiv/moltox21/molpcba)')
    parser.add_argument('--distill_set', type=str, default='train',
                        help='dataset set train/valid/test')
        # ---- Interaction
    parser.add_argument('--int_tau', type=float, default=1.,
                        help='temperature used during interaction')
    parser.add_argument('--int_epoch', type=int, default=100,
                        help='student training on real label, >500 is early stopping')
    parser.add_argument('--es_epochs', type=int, default=3,
                        help='consecutive how many epochs non-increase')
    parser.add_argument('--int_lr', type=float, default=1e-3,
                        help='learning rate for student on task during interaction')
    parser.add_argument('--int_optim', type=str, default='adam',
                        help='optimizer type during distillation, adam or adamW or sgd')
    parser.add_argument('--int_sched', type=eval, default=True,
                        help='Whether to use cosine scheduler')    
        # ---- Generate teacher
    parser.add_argument('--copy_what', type=str, default='best',
                        help='use the best or last epoch teacher in distillation')
    
    # ===== Wandb and saving results ====
    parser.add_argument('--run_name',default='test',type=str)
    parser.add_argument('--proj_name',default='P4_paper', type=str)
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
    if args.distill_data is not None:
        dis_loader = build_dataset(args,force_name=args.distill_data)
        dis_loader = dis_loader[args.distill_set]
    else:
        dis_loader = None
    task_loaders = build_dataset(args)
    
    for gen in range(args.generations):
        # =========== Step0: new agent
        if args.init_strategy == 'nil':
            student = get_init_net(args)
        elif args.init_strategy == 'mile':
            if gen > 1:
                student = old_teacher
            else:
                student = get_init_net(args)
        else:
            student = get_init_net(args)
            
        if args.dis_optim.lower()=='adam':
            optimizer_dis = optim.Adam(student.parameters(), lr=args.dis_lr)
        elif args.dis_optim.lower()=='adamw':
            optimizer_dis = optim.AdamW(student.parameters(), lr=args.dis_lr, weight_decay=0.01)
        elif args.dis_optim.lower()=='sgd':
            optimizer_dis = optim.SGD(student.parameters(), momentum=0.9, lr=args.dis_lr, weight_decay=0.01)
        
        if args.int_optim.lower()=='adam':
            optimizer_int = optim.Adam(student.parameters(), lr=args.int_lr)
        elif args.int_optim.lower()=='adamw':
            optimizer_int = optim.AdamW(student.parameters(), lr=args.int_lr, weight_decay=0.01)
        elif args.int_optim.lower()=='sgd':
            optimizer_int = optim.SGD(student.parameters(), momentum=0.9, lr=args.int_lr, weight_decay=0.01)
        if args.int_sched:
            if args.int_epoch>500:  # Now early stop
                tmax = 100
            else:
                tmax = args.int_epoch
            scheduler_int = optim.lr_scheduler.CosineAnnealingLR(optimizer_int,T_max=tmax,eta_min=1e-5)
        else:
            scheduler_int = optim.lr_scheduler.CosineAnnealingLR(optimizer_int,T_max=100,eta_min=args.int_lr)
        
        # =========== Step1: distillation, skip in first gen
        if gen > 0:
            train_distill(args, student, teacher, task_loaders['train'], dis_loader, optimizer_dis)
            old_teacher = copy.deepcopy(teacher)        
        # =========== Step2: solve task, track best valid acc
        best_vroc, best_v_ep, best_testroc, vacc_list = 0, 0, 0, []
        for epoch in range(args.int_epoch):
            train_task(args, student, task_loaders['train'], optimizer_int)
            scheduler_int.step()
            valid_roc = evaluate(args, student, task_loaders['valid'])
            test_roc = evaluate(args, student, task_loaders['test'])
            wandb.log({'Inter_val_roc':valid_roc})
            wandb.log({'Inter_test_roc':test_roc})
            vacc_list.append(valid_roc)
            if valid_roc > best_vroc:
                best_vroc = valid_roc
                best_testroc = test_roc
                best_v_ep = epoch
                if args.copy_what=='best':
                    teacher = copy.deepcopy(student)
            wandb.log({'best_val_epoch':best_v_ep})
            # ------- Early stop the FT if 3 non-increasing epochs
            if args.int_epoch>500 and early_stop_meets(vacc_list, best_vroc, how_many=args.es_epochs):
                break
        teacher = copy.deepcopy(student)
            
        #wandb.log({'End_gen_valid_roc':valid_roc})
        #wandb.log({'End_gen_test_roc':test_roc})
        #wandb.log({'Best_gen_valid_roc':best_vroc})
        #wandb.log({'Best_gen_test_roc':best_testroc})
    wandb.finish()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.config_file is not None:
        config = toml.load(os.path.join("configs",args.config_file+".toml"))
        args = update_args(args, config)
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    args.int_epoch = 0
    main(args)





  
