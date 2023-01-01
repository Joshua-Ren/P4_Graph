# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 21:43:20 2022

@author: YIREN
"""

import wandb
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from tqdm import tqdm
from utils.datasets import build_dataset
from utils.general import wandb_init, get_init_net, rnd_seed
from utils.nil_related import *
from ogb.graphproppred import Evaluator
import torch.optim as optim
from torch.nn.functional import cosine_similarity
from utils.general import AverageMeter

cls_criterion = torch.nn.BCEWithLogitsLoss()
Ce = torch.nn.CrossEntropyLoss()
Mse = torch.nn.MSELoss()

def train_epoch(args, model, optimizer, data_loader):
    losses = AverageMeter()
    model.train()
    for i,(x,y,reg,idx) in enumerate(data_loader):
        x, reg = x.float().cuda(), reg.float().cuda()
        reg = reg.unsqueeze(1)
        msg_all, h_all = model(x)
        optimizer.zero_grad()
        loss = nn.MSELoss(reduction='mean')(h_all,reg)
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), y.size(0))
        wandb.log({'Inter_loss':loss.data.item()})
    return losses.avg

def train_distill(args, student, teacher, optimizer, dataloader):
    losses = AverageMeter()
    teacher.eval()
    student.train()
    for i,(x,_,reg,idx) in enumerate(dataloader):
        x = x.float().cuda()
        msg_stud, _ = student(x)
        msg_teach, _ = teacher(x)   
        h1, h2, h3, h4 = msg_stud[:,0], msg_stud[:,1], msg_stud[:,2], msg_stud[:,3]
        optimizer.zero_grad()
        if args.dis_loss=='cesample':
            s1 = torch.distributions.categorical.Categorical(nn.Softmax(-1)(msg_teach[:,0]/args.dis_tau))
            s2 = torch.distributions.categorical.Categorical(nn.Softmax(-1)(msg_teach[:,1]/args.dis_tau))
            s3 = torch.distributions.categorical.Categorical(nn.Softmax(-1)(msg_teach[:,2]/args.dis_tau))
            s4 = torch.distributions.categorical.Categorical(nn.Softmax(-1)(msg_teach[:,3]/args.dis_tau))
            y1, y2, y3, y4 = s1.sample().long(), s2.sample().long(), s3.sample().long(), s4.sample().long()
            loss = Ce(h1,y1)+Ce(h2,y2)+Ce(h3,y3)+Ce(h4,y4)
        elif args.dis_loss=='argmax':
            y1, y2 = torch.argmax(msg_teach[:,0],dim=1), torch.argmax(msg_teach[:,1],dim=1)
            y3, y4 = torch.argmax(msg_teach[:,2],dim=1), torch.argmax(msg_teach[:,3],dim=1)
            loss = Ce(h1,y1)+Ce(h2,y2)+Ce(h3,y3)+Ce(h4,y4)
        elif args.dis_loss=='mse':
            loss = Mse(msg_stud, msg_teach)
        else:
            print('dis_loss must be cesample, argmax, or mse')
            return
        loss.backward()
        optimizer.step()
        wandb.log({'Distill_loss':loss.data.item()})
        losses.update(loss.data.item(), x.size(0))
    return losses.avg

def evaluate(args, model, dataloader):
    losses = AverageMeter()
    model.eval()
    with torch.no_grad(): 
        for i,(x,y,reg,idx) in enumerate(dataloader):
            x, reg = x.float().cuda(), reg.float().cuda()
            reg = reg.unsqueeze(1)
            msg_all, h_all = model(x)
            loss = nn.MSELoss(reduction='mean')(h_all,reg)
            wandb.log({'Test_loss':loss.data.item()})
            losses.update(loss.data.item(), reg.size(0))
        return losses.avg














