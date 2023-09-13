import wandb
import argparse
import numpy as np
import os
import copy
import random
import torch.optim as optim
import toml
from enegine_domainnet_celeba import train_epoch, train_distill, evaluate
from utils.general import update_args, wandb_init, get_init_net_domnet, rnd_seed, AverageMeter, early_stop_meets
from utils.nil_related import *
from utils.datasets_domainnet_celeba import generate_celeba_loader

def get_args_parser():
    # Training settings
    # ======= Usually default settings
    parser = argparse.ArgumentParser(description='DomainNet_celeba')
    parser.add_argument('--config_file', type=str, default=None,
                        help='the name of the toml configuration file')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--WD_ID',default='joshuaren', type=str,
                        help='W&D ID, joshuaren or joshua_shawn')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    # ======== Dataset and task related
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size of train and test set')
    parser.add_argument('--num_class', default=40, type=int,
                        help='40 for celeba, 203 for domainnet')    
    # ======== Model structure
    parser.add_argument('--model_pretrain', type=str, default="yes",
                        help='no, official or ...')
    parser.add_argument('--model_structure', type=str, default='direct',
                        help='Standard or sem or direct')
    parser.add_argument('--L', type=int, default=20,
                        help='No. word in SEM')
    parser.add_argument('--V', type=int, default=100,
                        help='word size in SEM')    
    
    # ======== Learning related
    parser.add_argument('--init_strategy', type=str, default='mile',
                        help='How to generate new student, nil or mile')
    parser.add_argument('--generations', default=2, type=int,
                        help='how many generations we train')

        # ---- Interaction
    parser.add_argument('--int_lr', default=1e-3, type=float,
                        help='learning rate used in interaction')  
    parser.add_argument('--int_epochs', default=2, type=int,
                        help='how many epochs we interact')
        # ---- Distillation
    parser.add_argument('--dis_lr', default=1e-4, type=float,
                        help='learning rate used in distillation')      
    parser.add_argument('--dis_epochs', default=2, type=int,
                        help='how many epochs we distill')
    parser.add_argument('--dis_loss', default='cesample', type=str,
                        help='the distillation loss: cesample, argmax, mse')
    parser.add_argument('--dis_tau', default=1, type=float,
                        help='tau used during cesample')
        # ---- Generate teacher
    parser.add_argument('--copy_what', type=str, default='best',
                        help='use the best or last epoch teacher in distillation')
    # ===== Wandb and saving results ====
    parser.add_argument('--run_name',default='test',type=str)
    parser.add_argument('--proj_name',default='P4_DomNet_directlabel', type=str)    
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
    if args.seed==0:
        args.seed = np.random.randint(1,10086)
    rnd_seed(args.seed)
    # ========== Prepare save folder and wandb ==========
    run_name = wandb_init(proj_name=args.proj_name, run_name=args.run_name, config_args=args)
    args.save_path = os.path.join('results','celeba',run_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # ========== Prepare the loader and optimizer
    train_loader, test_loader = generate_celeba_loader(args)
    
    for gen in range(args.generations):
        # =========== Step0: new agent
        if args.init_strategy == 'nil':
            student = get_init_net_domnet(args)
        elif args.init_strategy == 'mile':
            if gen > 1:
                student = old_teacher
            else:
                student = get_init_net_domnet(args)
        else:
            student = get_init_net_domnet(args)
        # ========= Distillation
        if gen>0:
            optimizer_dis = optim.SGD(student.parameters(), lr=args.dis_lr, momentum=0.9, weight_decay=5e-4,nesterov=True)
            scheduler_dis = optim.lr_scheduler.CosineAnnealingLR(optimizer_dis,T_max=args.dis_epochs,eta_min=1e-5)
            for i in range(args.dis_epochs):
                dis_loss = train_distill(args, student, teacher, optimizer_dis, train_loader)
                scheduler_dis.step()
            old_teacher = copy.deepcopy(teacher)   
        
        # ========= Interaction
        best_vacc = 0
        optimizer_inter = optim.SGD(student.parameters(), lr=args.int_lr, momentum=0.9, weight_decay=5e-4,nesterov=True)
        #optimizer_inter = optim.Adam(student.parameters(), lr=args.int_lr, weight_decay=5e-4)
        scheduler_inter = optim.lr_scheduler.CosineAnnealingLR(optimizer_inter,T_max=args.int_epochs,eta_min=1e-5)        
        for i in range(args.int_epochs):
            wandb.log({'Train_Epoch':i})
            loss = train_epoch(args, student, optimizer_inter, train_loader)
            scheduler_inter.step()    
            vacc = evaluate(args, student, test_loader)
            wandb.log({'Valid_acc':vacc})
            if vacc > best_vacc:
                best_vacc = vacc
                if args.copy_what=='best':
                    teacher = copy.deepcopy(student)
        wandb.log({'Best_vacc':best_vacc})
        if args.copy_what=='last':
            teacher = copy.deepcopy(student)
    wandb.finish()
    
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if args.config_file is not None:
        config = toml.load(os.path.join("configs",args.config_file+".toml"))
        args = update_args(args, config)
    if args.model_structure=="direct":
        args.dis_loss='direct_label_sample'
    #args.model_structure="sem"
    main(args)
    """
    args.model_structure="sem"
    model = get_init_net_domnet(args)
    train_l, val_l = generate_celeba_loader(args)
    from utils.datasets_domainnet import label_mapping
    for i,(x,y) in enumerate(train_l):
        x = x.cuda()
        y = y.float().cuda()
        break
    msg, h_all = model(x)
    Bce = torch.nn.BCELoss()
    Sig = torch.nn.Sigmoid()
    """
    
    
    
    
    
    
    