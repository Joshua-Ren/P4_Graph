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
from ogb.lsc import PCQM4Mv2Evaluator

cls_criterion = torch.nn.BCEWithLogitsLoss()
ce_criterion = torch.nn.CrossEntropyLoss()
#reg_criterion = torch.nn.MSELoss()
reg_criterion = torch.nn.L1Loss()
# ================== Different training stages =======================
def train_task(args, model, loader, optimizer):
    # --------- Update the whole network under the task-supervision
    # Can be used in generating the baseline, or as a phase in NIL
    # Wandb records: 
    model.train()
    if args.dataset_name=='pcqm':
        evaluator = PCQM4Mv2Evaluator()
    else:
        evaluator = Evaluator(args.dataset_name)
    
    ### Only for subtask
    if args.dataset_forcetask==1:
        evaluator = Evaluator('ogbg-molhiv')

    y_true, y_pred = [], []
    for step, batch in enumerate(loader):
        batch = batch.to(args.device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            _, pred = model.task_forward(batch, args.int_tau)           
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if is_labeled.sum()==0:
                continue
            if "classification" in args.task_type: 
                output, target = pred[is_labeled], batch.y.to(torch.float32)[is_labeled]
                loss = cls_criterion(output, target)
                #loss = torchvision.ops.sigmoid_focal_loss(output, target).sum()
            else:
                loss = reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled].unsqueeze(-1))
            loss.backward()
            optimizer.step()
            wandb.log({'Inter_loss':loss.data.item()})
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    if args.dataset_name=='pcqm':
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    eval_result = evaluator.eval(input_dict)
    wandb.log({'Inter_train_roc':eval_result[args.eval_metric]})
    
def train_distill(args, student, teacher, task_loader, unsup_loader, optimizer):
    # ------------ Train the student using the teacher's prediction (argmax, sample, mse)
    # Wandb record: distill_loss
    #               msg_overlap, the overlap ratio of messages between teacher and student
    # Controlled by args.steps_dis
    teacher.eval()
    student.train()
    cnt = 0
    task_iter = iter(task_loader)
    if unsup_loader is not None:
        unsup_iter = iter(unsup_loader)
    while True:
        if unsup_loader is not None and np.random.randint(2): 
            try:
                batch = next(unsup_iter)
            except:
                unsup_iter = iter(unsup_loader)
                batch = next(unsup_iter)
        else:
            try:
                batch = next(task_iter)
            except:
                task_iter = iter(task_loader)
                batch = next(task_iter)
        cnt += 1
        if cnt > args.dis_steps:
            return cnt
        tmp_batchsize = batch.y.shape[0]
        batch = batch.to(args.device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            teach_logits, _ = teacher.distill_forward(batch)
            stud_logits, _ = student.distill_forward(batch)
            teach_logits, stud_logits = teach_logits/args.dis_tau, stud_logits
            if args.dis_loss == 'ce_argmax':
                teach_label = teach_logits.argmax(-1)
                loss = ce_criterion(stud_logits.reshape(-1,args.V),teach_label.reshape(-1,))
            elif args.dis_loss == 'ce_sample':
                sampler = torch.distributions.categorical.Categorical(nn.Softmax(-1)(teach_logits))
                teach_label = sampler.sample().long()
                loss = ce_criterion(stud_logits.reshape(-1,args.V),teach_label.reshape(-1,))
            elif args.dis_loss == 'noisy_ce_sample':
                epsilon = torch.randn_like(teach_logits)
                dist = F.softmax((teach_logits + epsilon), -1)
                sampler = torch.distributions.categorical.Categorical(dist)
                teach_label = sampler.sample().long()
                loss = ce_criterion(stud_logits.reshape(-1,args.V),teach_label.reshape(-1,))
            elif args.dis_loss == 'mse':
                loss = nn.MSELoss(reduction='mean')(stud_logits, teach_logits)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({'Dis_loss':loss.data.item()})
            msg_dist = cal_msg_distance_logits(stud_logits,teach_logits)
            wandb.log({'Dis_msg_overlap':msg_dist.item()})

# ================== Different evaluation methods during training
def evaluate(args, model, loader):
    #---- Basic evaluation, give the loader and return rocauc or other metrics
    if args.dataset_name=='pcqm':
        evaluator = PCQM4Mv2Evaluator()
    else:
        evaluator = Evaluator(args.dataset_name)

    ### Only for subtask
    if args.dataset_forcetask==1:
        evaluator = Evaluator('ogbg-molhiv')

    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        batch = batch.to(args.device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                is_labeled = batch.y == batch.y
                if is_labeled.sum()==0:
                    continue
                _, pred = model.task_forward(batch, args.int_tau)
                y_true.append(batch.y.view(pred.shape).detach().cpu())
                y_pred.append(pred.detach().cpu())
            
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    if args.dataset_name=='pcqm':
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze() 
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    eval_result = evaluator.eval(input_dict)
    rocauc = eval_result[args.eval_metric]
    return rocauc