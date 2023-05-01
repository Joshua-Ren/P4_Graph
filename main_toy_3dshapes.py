# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 12:59:45 2022

@author: YIREN
"""
import wandb
import argparse
import numpy as np
import os
import copy
import torch.optim as optim
import toml
from enegine_toy_3dshapes import train_epoch, train_distill, evaluate
from utils.general import update_args, wandb_init, get_init_net_toy, rnd_seed, AverageMeter, early_stop_meets
from utils.nil_related import *
from utils.toy_example import generate_3dshape_loaders

def get_args_parser():
    # Training settings
    # ======= Usually default settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--config_file', type=str, default=None,
                        help='the name of the toml configuration file')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--WD_ID',default='joshuaren', type=str,
                        help='W&D ID, joshuaren or joshua_shawn')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    # ======== Dataset and task related
    parser.add_argument('--dataset_name', default='3dshape', type=str,
                        help='3dshapes or dsprite')    
    parser.add_argument('--sup_ratio', default=0.1, type=float,
                        help='ratio of the training factors')
    parser.add_argument('--batch_size', default=20, type=int,
                        help='batch size of train and test set')
    parser.add_argument('--num_class', default=1, type=int,
                        help='How many reg-tasks, 1~6')
    parser.add_argument('--data_per_g', default=1, type=int,
                        help='how many samples for each G')

    
    # ======== Model structure
    parser.add_argument('--model_structure', type=str, default='standard',
                        help='Standard or sem')
    parser.add_argument('--L', type=int, default=4,
                        help='No. word in SEM')
    parser.add_argument('--V', type=int, default=10,
                        help='word size in SEM')    
    
    # ======== Learning related
    parser.add_argument('--init_strategy', type=str, default='nil',
                        help='How to generate new student, nil or mile')
    parser.add_argument('--generations', default=5, type=int,
                        help='how many generations we train')
    parser.add_argument('--lr_min', default=1e-5, type=float,
                        help='cosine decay to this learning rate')
    parser.add_argument('--bob_adapt_ep', default=20, type=float,
                        help='how many epoch we adapt bob first')    

        # ---- Interaction
    parser.add_argument('--int_lr', default=1e-3, type=float,
                        help='learning rate used in interaction')  
    parser.add_argument('--int_epochs', default=250, type=int,
                        help='how many epochs we interact')
        # ---- Distillation
    parser.add_argument('--dis_lr', default=1e-3, type=float,
                        help='learning rate used in distillation')      
    parser.add_argument('--dis_epochs', default=50, type=int,
                        help='how many epochs we distill')
    parser.add_argument('--dis_loss', default='cesample', type=str,
                        help='the distillation loss: cesample, argmax, mse')
    parser.add_argument('--dis_dataset', default='train', type=str,
                        help='the distillation loss: train, test, unsup')
    parser.add_argument('--dis_tau', default=1, type=float,
                        help='tau used during cesample')
        # ---- Generate teacher
    parser.add_argument('--copy_what', type=str, default='best',
                        help='use the best or last epoch teacher in distillation')
    # ===== Wandb and saving results ====
    parser.add_argument('--run_name_seed',default='test',type=str)
    parser.add_argument('--proj_name',default='P4_toy', type=str)    
    return parser

def main(args):
    # Model and optimizers are build in
    # In each generation:
    #   Step0: prepare everything
    #   Step1: distillation, skip if first gen
    #   [Step2: student SSL like SimCLR]
    #   Step2: student ft on task
    #   Step3: student becomes the teacher
    # ========== Generate seed ==========
    results = {'tloss':[],'vloss':[], 'dis_loss':[]}
    if args.seed==0:
        args.seed = np.random.randint(1,10086)
    rnd_seed(args.seed)
    # ========== Prepare save folder and wandb ==========
    run_name = wandb_init(proj_name=args.proj_name, run_name=args.run_name, config_args=args)
    args.save_path = os.path.join('results','toy_example',run_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # ========== Prepare the loader and optimizer
    train_loader, test_loader, unsup_loader = generate_3dshape_loaders(args)
    if args.dis_dataset=='train':
        dis_loader = train_loader
    elif args.dis_dataset=='test':
        dis_loader = test_loader
    elif args.dis_dataset=='unsup':
        dis_loader = unsup_loader
    
    for gen in range(args.generations):
        # =========== Step0: new agent
        if args.init_strategy == 'nil':
            student = get_init_net_toy(args)
        elif args.init_strategy == 'mile':
            if gen > 1:
                student = old_teacher
            else:
                student = get_init_net_toy(args)
        else:
            student = get_init_net_toy(args)
        # ========= Distillation
        optimizer_inter = optim.SGD(student.parameters(), lr=args.dis_lr, momentum=0.9, weight_decay=5e-4,nesterov=True)
        if gen>0:
            optimizer_dis = optim.SGD(student.parameters(), lr=args.int_lr, momentum=0.9, weight_decay=5e-4,nesterov=True)
            for i in range(args.dis_epochs):
                wandb.log({'idx_epoch':i})
                dis_loss = train_distill(args, student, teacher, optimizer_dis, dis_loader)
                results['dis_loss'].append(dis_loss)
            old_teacher = copy.deepcopy(teacher)   
        # ========= Interaction
        best_vloss = 10
            # --- Bob adaptation
        bob_optim = optim.SGD(student.Bob.parameters(), lr=args.dis_lr, momentum=0.9, weight_decay=5e-4,nesterov=True)
        for i in range(args.bob_adapt_ep):
            train_epoch(args, student, bob_optim, train_loader)
        for i in range(args.int_epochs):
            wandb.log({'idx_epoch':i})
            loss = train_epoch(args, student, optimizer_inter, train_loader)
            if i%5==0 or i==args.int_epochs-1:
                vloss = evaluate(args, student, test_loader)
                results['tloss'].append(loss)
                results['vloss'].append(vloss)
                wandb.log({'train_loss':loss})
                wandb.log({'test_loss':vloss})
                if vloss < best_vloss:
                    best_vloss = vloss
                    if args.copy_what=='best':
                        teacher = copy.deepcopy(student)
        wandb.log({'Best_vloss':best_vloss})
        if args.copy_what=='last':
            teacher = copy.deepcopy(student)
    wandb.log({'Report_loss':best_vloss})
    wandb.finish()
    result_save_name = os.path.join(args.save_path, 'loss.npy')
    np.save(result_save_name, results)
    # use xxx.item().get(key) to extract
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.config_file is not None:
        config = toml.load(os.path.join("configs",args.config_file+".toml"))
        args = update_args(args, config)
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    #main(args)
    
    # ==== Long experiments ====
    ALPHAS = [0.8, 0.5, 0.2, 0.1, 0.02, 0.002]# [0.002, 0.02, 0.1, 0.2, 0.5, 0.8]
    SEEDS = [1024, 10086, 42, 1314]
    for seed in SEEDS:
        for alpha in ALPHAS:
            args.seed = seed
            args.sup_ratio = alpha
            args.run_name = args.run_name_seed + '_alpha_'+str(alpha) +'_seed_'+str(seed)
            main(args)