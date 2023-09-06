# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 23:42:29 2023

@author: YIREN
"""

import wandb
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from tqdm import tqdm
from utils.general import wandb_init, get_init_net, rnd_seed
#from utils.nil_related import *
#from ogb.graphproppred import Evaluator
import torch.optim as optim
from torch.nn.functional import cosine_similarity
from utils.general import AverageMeter
#from utils.datasets_domainnet import label_mapping

Bce_log = torch.nn.BCEWithLogitsLoss()
Ce = torch.nn.CrossEntropyLoss()
Mse = torch.nn.MSELoss()
Bce = torch.nn.BCELoss()
Sig = torch.nn.Sigmoid()

def train_epoch(args, model, optimizer, data_loader):
    losses = AverageMeter()
    model.train()     
    acc = AverageMeter()
    for i,(x,y) in enumerate(data_loader):
        x, y = x.float().cuda(), y[:args.num_class].float().cuda()
        msg_all, h_all = model(x)
        optimizer.zero_grad()
        loss = Bce(Sig(h_all),y)
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), y.size(0))
        wandb.log({'Inter_loss':loss.data.item()})
        
        pred = (Sig(h_all)>0.5)==(y>0.5)
        tmp_acc = pred.sum()/(y.shape[0]*y.shape[1])
        acc.update(tmp_acc.item())
        wandb.log({'Train_acc':acc.avg})
    return losses.avg

def train_distill(args, student, teacher, optimizer, dataloader):
    losses = AverageMeter()
    teacher.eval()
    student.train()
    for i,(x,_) in enumerate(dataloader):
        x = x.float().cuda()
        teach_logits, _ = teacher(x)   
        stud_logits, _ = student(x)
        optimizer.zero_grad()
        if args.dis_loss=='argmax':
            teach_label = teach_logits.argmax(-1)
            loss = Ce(stud_logits.reshape(-1,args.V),teach_label.reshape(-1,))
        elif args.dis_loss=='cesample':
            sampler = torch.distributions.categorical.Categorical(nn.Softmax(-1)(teach_logits))
            teach_label = sampler.sample().long()
            loss = Ce(stud_logits.reshape(-1,args.V),teach_label.reshape(-1,))
        elif args.dis_loss=='mse':
            loss = Mse(stud_logits, teach_logits)
        else:
            print('dis_loss must be cesample, argmax, or mse')
            return
        loss.backward()
        optimizer.step()
        wandb.log({'Distill_loss':loss.data.item()})
        losses.update(loss.data.item(), x.size(0))
    return losses.avg

def evaluate(args, model, dataloader):
    acc = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i,(x,y) in enumerate(dataloader):
            x, y = x.float().cuda(), y[:args.num_class].float().cuda()
            msg_all, h_all = model(x)
            pred = (Sig(h_all)>0.5)==(y>0.5)
            tmp_acc = pred.sum()/(y.shape[0]*y.shape[1])
            acc.update(tmp_acc.item())
        return acc.avg

"""
def train_epoch(args, model, optimizer, data_loader):
    losses = AverageMeter()
    model.train()     
    acc1 = AverageMeter()
    acc2 = AverageMeter()
    for i,(x,y) in enumerate(data_loader):
        y1, y2 = label_mapping(y)
        x, y1, y2 = x.float().cuda(), y1.long().cuda(), y2.long().cuda()
        msg_all, h_all = model(x)
        optimizer.zero_grad()
        loss = Ce(h_all[:,:200],y1) + Ce(h_all[:,200:],y2)
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), y.size(0))
        wandb.log({'Inter_loss':loss.data.item()})
        
        pred1 = h_all[:,:200].argmax(-1)
        pred2 = h_all[:,200:].argmax(-1)
        acc1.update(((pred1==y1).sum()/y1.shape[0]).item())
        acc2.update(((pred2==y2).sum()/y1.shape[0]).item())
        wandb.log({'Train_acc1':acc1.avg})
        wandb.log({'Train_acc2':acc2.avg})
    return losses.avg

def train_distill(args, student, teacher, optimizer, dataloader):
    losses = AverageMeter()
    teacher.eval()
    student.train()
    for i,(x,_) in enumerate(dataloader):
        x = x.float().cuda()
        teach_logits, _ = teacher(x)   
        stud_logits, _ = student(x)
        optimizer.zero_grad()
        if args.dis_loss=='cesample':
            teach_label = teach_logits.argmax(-1)
            loss = Ce(stud_logits.reshape(-1,args.V),teach_label.reshape(-1,))
        elif args.dis_loss=='argmax':
            sampler = torch.distributions.categorical.Categorical(nn.Softmax(-1)(teach_logits))
            teach_label = sampler.sample().long()
            loss = Ce(stud_logits.reshape(-1,args.V),teach_label.reshape(-1,))
        elif args.dis_loss=='mse':
            loss = Mse(stud_logits, teach_logits)
        else:
            print('dis_loss must be cesample, argmax, or mse')
            return
        loss.backward()
        optimizer.step()
        wandb.log({'Distill_loss':loss.data.item()})
        losses.update(loss.data.item(), x.size(0))
    return losses.avg

def evaluate(args, model, dataloader):
    acc1 = AverageMeter()
    acc2 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i,(x,y) in enumerate(dataloader):
            y1, y2 = label_mapping(y)
            x, y1, y2 = x.float().cuda(), y1.long().cuda(), y2.long().cuda()
            msg_all, h_all = model(x)
            pred1 = h_all[:,:200].argmax(-1)
            pred2 = h_all[:,200:].argmax(-1)
            acc1.update(((pred1==y1).sum()/y1.shape[0]).item())
            acc2.update(((pred2==y2).sum()/y1.shape[0]).item())
        return acc1.avg, acc2.avg


"""

    
    