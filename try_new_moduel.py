# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 11:17:24 2022

@author: YIREN
"""


from engine_phases import *
from utils.datasets import *
from utils.general import *
from utils.nil_related import *
from engine_phases import *
import torch.optim as optim
import torch
import argparse
import numpy as np
import random
import os

def get_args_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset_name', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--ssl_lr', type=float, default=1e-3,
                        help='learning rate for student')
    parser.add_argument('--lp_lr', type=float, default=1e-3,
                        help='learning rate for student')
    parser.add_argument('--ft_lr', type=float, default=1e-3,
                        help='learning rate for student')
    parser.add_argument('--byol_eta', type=float, default=0.99,
                        help='eta in EMA')  
    # ==== SEM settings ======
    parser.add_argument('--model_type', type=str, default='gcn',
                        help='gcn, gin, gcn-virtual, gin-virtual')  
    parser.add_argument('--L', type=int, default=200,
                        help='No. word in SEM')
    parser.add_argument('--V', type=int, default=20,
                        help='word size in SEM')
    parser.add_argument('--ssl_tau', type=float, default=1.,
                        help='temperature in SEM')
    parser.add_argument('--dis_sem_tau', type=float, default=1.,
                        help='temperature in SEM')
    parser.add_argument('--ft_tau', type=float, default=1.,
                        help='temperature in SEM')
    # ===== NIL settings ======
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
    # ===== Distillation settings ======
    parser.add_argument('--dis_loss', type=str, default='ce_argmax',
              help='how the teacher generate the samples, ce_argmax, ce_sample, mse, kld')
    parser.add_argument('--dis_smp_tau', type=float, default=1.,
              help='temperature used when teacher generating sample, 0 is argmax')
    parser.add_argument('--dis_optim_type', type=str, default='sgd',
              help='optimizer used during distillation, sgd or adam')
    parser.add_argument('--dis_lr', type=float, default=1e-3,
                        help='learning rate for student')
    parser.add_argument('--dis_wd', type=float, default=0.,
                        help='weight decay used in distillation')
    # ===== SSL settings ======
    parser.add_argument('--inter_alpha', type=float, default=0,
                        help='balance between task loss and inter-loss, 0 is all inter')
    parser.add_argument('--byol_loss', type=str, default='mse',
                        help='loss type used in byol, mse or cosine')
    # ===== Wandb and saving results ====
    parser.add_argument('--run_name',default='test',type=str)
    parser.add_argument('--proj_name',default='P4_SSL_Graph_new', type=str)
    parser.add_argument('--save_dir', default=None,
                        help='path of the pretrained checkpoint')    
    return parser

args = get_args_parser()
args = args.parse_args()
args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

loaders = build_dataset(args)
tmp_model = get_init_net(args)
load_path = os.path.join('results','GCN_baseline.pth')
tmp_model.load_state_dict(torch.load(load_path),strict=True)
wandb.init()
train_roc, valid_roc, test_roc = eval_all(args, tmp_model, loaders, title='Stud_')










