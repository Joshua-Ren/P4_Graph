import wandb
import copy
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from tqdm import tqdm
from utils.datasets import build_dataset
from utils.general import wandb_init, get_init_net, rnd_seed, AverageMeter
from utils.nil_related import *
from ogb.graphproppred import Evaluator
import torch.optim as optim
from torch.nn.functional import cosine_similarity

cls_criterion = torch.nn.BCEWithLogitsLoss()
ce_criterion = torch.nn.CrossEntropyLoss()
reg_criterion = torch.nn.MSELoss()

# ================== Different training stages =======================
def train_task(args, model, loader, optimizer, scheduler=None, model0=None,):
    # --------- Update the whole network under the task-supervision
    # Can be used in generating the baseline, or as a phase in NIL
    # Wandb records: 
    
    ft_losses = AverageMeter()
    ft_msg_dists = AverageMeter()
    ft_msg_topsim = AverageMeter()
    ft_msg_entropy = AverageMeter()
    ft_train_roc = AverageMeter()
    model.train()
    evaluator = Evaluator(args.dataset_name)
    y_true, y_pred = [], []
    for step, batch in enumerate(loader):
        batch = batch.to(args.device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            logits, pred = model.task_forward(batch, args.ft_tau)
            if model0 is not None:
                # ----- Then we should report the message distance
                logits0, _ = model0.task_forward(batch, args.ft_tau)
                msg_dist = cal_msg_distance_logits(logits,logits0)
                ft_msg_dists.update(msg_dist.item())
                wandb.log({'ft_msg_drift':ft_msg_dists.avg})
            if True:    # Whether to calcualte the topsim
                corr, p = cal_topsim(logits, batch)
                entropy = cal_att_entropy(logits)
                ft_msg_entropy.update(entropy)
                ft_msg_topsim.update(corr)
                wandb.log({'ft_msg_entropy':ft_msg_entropy.avg})
                wandb.log({'ft_topsim':ft_msg_topsim.avg})                    
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in args.task_type: 
                output, target = pred[is_labeled], batch.y.to(torch.float32)[is_labeled]
                loss = cls_criterion(output, target)
                #loss = torchvision.ops.sigmoid_focal_loss(output, target).sum()
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            loss.backward()
            optimizer.step()
            ft_losses.update(loss.data.item(), batch.x.size(0))
            wandb.log({'ft_task_loss':ft_losses.avg})           
            # ------ Update train acc each batch
            y_true = batch.y.view(pred.shape).detach().cpu().numpy()
            y_pred = pred.detach().cpu().numpy()
            input_dict = {"y_true": y_true, "y_pred": y_pred}
            eval_result = evaluator.eval(input_dict)
            ft_train_roc.update(eval_result[args.eval_metric])
            wandb.log({'task_train_roc':ft_train_roc.avg})
            if scheduler is not None:
                scheduler.step()

    
def train_distill(args, student, teacher, loader, optimizer):
    # ------------ Train the student using the teacher's prediction (argmax, sample, mse)
    # Wandb record: distill_loss
    #               msg_overlap, the overlap ratio of messages between teacher and student
    dis_losses = AverageMeter()
    dis_msg_dists = AverageMeter()
    dis_msg_topsim = AverageMeter()
    dis_msg_entropy = AverageMeter()
    teacher.eval()
    student.train()
    for step, batch in enumerate(loader):
        tmp_batchsize = batch.y.shape[0]
        batch = batch.to(args.device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            teach_logits, tech_p = teacher.distill_forward(batch, args.dis_sem_tau)
            stud_logits, stud_p = student.distill_forward(batch, args.dis_sem_tau)
            
            if True:    # Whether to calcualte the topsim
                corr, p = cal_topsim(stud_logits, batch)
                entropy = cal_att_entropy(stud_logits)
                dis_msg_entropy.update(entropy)
                dis_msg_topsim.update(corr)
                wandb.log({'Dis_msg_entropy':dis_msg_entropy.avg})
                wandb.log({'Dis_topsim':dis_msg_topsim.avg})           
            
            if args.dis_loss == 'ce_argmax':
                teach_label = teach_logits.argmax(-1)
                loss = ce_criterion(stud_logits.reshape(-1,args.V),teach_label.reshape(-1,))
            elif args.dis_loss == 'ce_sample':
                sampler = torch.distributions.categorical.Categorical(nn.Softmax(-1)(teach_logits/args.dis_smp_tau))
                teach_label = sampler.sample().long()
                loss = ce_criterion(stud_logits.reshape(-1,args.V),teach_label.reshape(-1,))
            elif args.dis_loss == 'noisy_ce_sample':
                epsilon = torch.randn_like(teach_logits)
                dist = F.softmax((teach_logits + epsilon)/args.dis_smp_tau, -1)
                sampler = torch.distributions.categorical.Categorical(dist)
                teach_label = sampler.sample().long()
                loss = ce_criterion(stud_logits.reshape(-1,args.V),teach_label.reshape(-1,))
            elif args.dis_loss == 'mse':
                loss = nn.MSELoss(reduction='mean')(stud_logits, teach_logits)
            elif args.dis_loss == 'kld':   # Seems always giving nan, see what's the problem
                loss = nn.KLDivLoss(reduction='batchmean')(torch.log(stud_logits.reshape(-1,args.V)),nn.Softmax(-1)(teach_logits))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dis_losses.update(loss.data.item(), batch.x.size(0))
            wandb.log({'distill_loss':dis_losses.avg})
            msg_dist = cal_msg_distance_logits(stud_logits,teach_logits)
            dis_msg_dists.update(msg_dist.item())
            wandb.log({'Dis_msg_overlap':dis_msg_dists.avg})

def train_simclr(args, model, loader, optimizer):
    # 1. X --> aug(X1) and aug(X2)
    # 2. M1 = [GCN+SEM+MLP](X1)
    #    M2 = [GCN](X2)
    # 3. Calculate loss: 
    #    a. F = M1.T*M2 to get N*N matirx (diagnoal is positive pairs)
    #    b. CE-loss(F, [1,2,...,N]) + CE-loss(F.T, [1,2,...,N])
    losses = AverageMeter()
    model.train()
    for step, batch in enumerate(loader):
        tmp_batchsize = batch.y.shape[0]
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            continue
        else:
            batch1, batch2 = get_batch_aug(args, batch, aug_type='node_gaussian')
            batch1.to(args.device)
            batch2.to(args.device)
            # Not ready yet

def train_byol(args, online_model, target_model, loader,optimizer):
    # 1. X --> aug(X1) and aug(X2)
    # 2. onl_z = q(f(X1))
    #    tgt_z = f'(X2)
    #    loss = (1-a)*MSE_or_cosine(onl_z, tgt_z)+a*task_loss
    # 3. EMA update target_model
    # 4. Track eval_acc or svm_acc during playing
    losses = AverageMeter()
    online_model.train()
    target_model.eval()
    trans1 = get_graph_drop_transform(0.1, 0.1)
    trans2 = get_graph_drop_transform(0.1, 0.1)
    for step, batch in enumerate(loader):
        tmp_batchsize = batch.y.shape[0]
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            batch1, batch2 = trans1(batch), trans2(batch)
            batch1.to(args.device)
            batch2.to(args.device)
            batch.to(args.device)
            _, q_theta = online_model.ssl_forward(batch1, args.ssl_tau)
            z_theta, _ = target_model.ssl_forward(batch2, args.ssl_tau)
            z_theta = z_theta.detach()
      
        optimizer.zero_grad()
        loss_task = 0
        if args.inter_alpha>0:
            pred = online_model.task_forward(batch)
            # ----- Can also incorporate task-loss together
            is_labeled = batch.y == batch.y
            if "classification" in args.dataset.task_type: 
                output, target = pred[is_labeled], batch.y.to(torch.float32)[is_labeled]
                loss_task = cls_criterion(output, target)
            else:
                loss_task = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
        if args.byol_loss == 'mse':
            loss_byol = nn.MSELoss()(q_theta,z_theta)
        elif args.byol_loss == 'cosine':
            loss_byol = 2 - cosine_similarity(q_theta, z_theta.detach(), dim=-1).mean()
      
        loss = (1-args.inter_alpha)*loss_byol + args.inter_alpha*loss_task
        loss.backward()
        optimizer.step()
        losses.update(loss.data.item(), batch.x.size(0))
        wandb.log({'byol_loss':losses.avg})
        EMA_update(target_model, online_model, eta=args.byol_eta)

def train_bgrl(args, online_model, target_model, loader, optimizer):
    # Quite similar to byol, but z is the nodes level rather than pooled level
    # 1. X --> aug(X1) and aug(X2)
    # 2. onl_z = q(f(X1))
    #    tgt_z = f'(X2)
    #    loss = (1-a)*MSE_or_cosine(onl_z, tgt_z)+a*task_loss
    # 3. EMA update target_model
    # 4. Track eval_acc or svm_acc during playing
    pass


# ================== Different evaluation methods during training
def eval(args, model, loader):
    #---- Basic evaluation, give the loader and return rocauc or other metrics
  evaluator = Evaluator(args.dataset_name)
  model.eval()
  y_true = []
  y_pred = []

  for step, batch in enumerate(loader):
    batch = batch.to(args.device)
    if batch.x.shape[0] == 1:
      pass
    else:
      with torch.no_grad():
        _, pred = model.task_forward(batch)
      y_true.append(batch.y.view(pred.shape).detach().cpu())
      y_pred.append(pred.detach().cpu())
  y_true = torch.cat(y_true, dim = 0).numpy()
  y_pred = torch.cat(y_pred, dim = 0).numpy()
  input_dict = {"y_true": y_true, "y_pred": y_pred}
  eval_result = evaluator.eval(input_dict)
  rocauc = eval_result[args.eval_metric]
  return rocauc

def eval_all(args, student, loaders, title='Stud_', no_train=False):
    
  if no_train:
    valid_loader, test_loader = loaders['valid'], loaders['test']
    valid_roc = eval(args, student, valid_loader)
    test_roc = eval(args, student, test_loader)
    wandb.log({title+'valid_roc':valid_roc})
    wandb.log({title+'test_roc':test_roc})
    train_roc = 0
  else:
    train_loader, valid_loader, test_loader = loaders['train'], loaders['valid'], loaders['test']
    train_roc = eval(args, student, train_loader)
    valid_roc = eval(args, student, valid_loader)
    test_roc = eval(args, student, test_loader)
    wandb.log({title+'train_roc':train_roc})
    wandb.log({title+'valid_roc':valid_roc})
    wandb.log({title+'test_roc':test_roc})
  return train_roc, valid_roc, test_roc

def eval_probing(args, student, loaders, title='Distill_prob_', no_train=True):
  # ------ Fix the backbone and only optimize the task head
  model = copy.deepcopy(student)
  losses = AverageMeter()
  model.train()
  lp_optim = optim.Adam(model.task_head.parameters(),lr=args.lp_lr)
  for i in range(args.epochs_lp):
    for step, batch in enumerate(loaders['train']):
      batch = batch.to(args.device)
      if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
        pass
      else:
        _, pred = model.task_forward(batch, args.ft_tau)
        lp_optim.zero_grad()
        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batch.y == batch.y
        if "classification" in args.task_type: 
          output, target = pred[is_labeled], batch.y.to(torch.float32)[is_labeled]
          loss = cls_criterion(output, target)
          #loss = torchvision.ops.sigmoid_focal_loss(output, target).sum()
        else:
          loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
        loss.backward()
        lp_optim.step()
        losses.update(loss.data.item(), batch.x.size(0))
        wandb.log({title+'lp_loss':losses.avg})
    train_roc, valid_roc, test_roc = eval_all(args, model, loaders=loaders, 
                                              title=title, no_train=no_train)
  del model
  return train_roc, valid_roc, test_roc

def eval_svm(args, student, loaders, title='Distill_svm_', no_train=True):
  # ------ Fix the backbone and use SVM to measure task-performance
    pass

def eval_topsim(args, student, loaders, title='Distill_topsim_'):
    pass


