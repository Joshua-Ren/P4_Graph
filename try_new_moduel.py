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
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt

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
    parser.add_argument('--drop_ratio', type=float, default=0.5,
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
    parser.add_argument('--num_layer', type=int, default=3,
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

args = get_args_parser()
args = args.parse_args()
args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")


loaders = build_dataset(args)
teacher = get_init_net(args)
'''
loaders = build_dataset(args)
teacher = get_init_net(args)
student = copy.deepcopy(teacher)
for step, batch in enumerate(loaders['train']):
    break


for n,p1 in teacher.task_head.named_parameters():
    break
for n,p2 in student.task_head.named_parameters():
    break
p1==p2
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

teacher.task_head.apply(init_weights)



exp_name = args.backbone_type+'_'+args.bottle_type
load_path = os.path.join('results',exp_name,args.dataset_name,exp_name+'_'+args.dataset_name+'.pth')
online.load_state_dict(torch.load(load_path),strict=True)

logits0, pred0 = online.task_forward(batch.cuda())

entropy = cal_att_entropy(logits0)




msg_dists = []
y_dists = []
for i in range(msg0.shape[0]):
    for j in range(i):
        msg_dists.append((msg0[i] == msg0[j]).sum().cpu())
        y_dists.append((batch.y[i] == batch.y[j]).sum().cpu())

#from scipy import stats
#corr,p = stats.spearmanr(msg_dists,y_dists)
#print(corr,p)
#plt.scatter(msg_dists,y_dists)


#_, q_out = online.refgame_forward(batch.cuda())
#z_out, _ = target.refgame_forward(batch.cuda())
#corr_matrix = torch.mm(q_out,z_out.T)
#tgt_vector = torch.arange(0,z_out.shape[0],1).long().to(args.device)
#loss_byol = nn.CrossEntropyLoss()(corr_matrix,tgt_vector)
#loss = 2 - cosine_similarity(hq_node, h_node.detach(), dim=-1).mean()
#load_path = os.path.join('results','GCN_baseline.pth')
#tmp_model.load_state_dict(torch.load(load_path),strict=True)
#wandb.init()
#train_roc, valid_roc, test_roc = eval_all(args, tmp_model, loaders, title='Stud_')
'''







