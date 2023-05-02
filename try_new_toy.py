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
from utils.toy_example import get_dataloaders
from models.vae import BetaVAE_H
import shutil
from PIL import Image
from matplotlib import pyplot as plt

def get_args_parser():
    # Training settings
    # ======= Usually default settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--config_file', type=str, default=None,
                        help='the name of the toml configuration file')
    parser.add_argument('--seed', default=10086, type=int)
    parser.add_argument('--WD_ID',default='joshuaren', type=str,
                        help='W&D ID, joshuaren or joshua_shawn')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    # ======== Dataset and task related
    parser.add_argument('--dataset_name', default='mpi3d', type=str,
                        help='3dshapes or dsprites, mpi3d')    
    parser.add_argument('--sup_ratio', default=0.2, type=float,
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
    parser.add_argument('--run_name',default='test',type=str)
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
    # ========== Prepare the loader and optimizer

def latent_to_index(latents):
    latent_sizes = np.array([ 1,  6,  6, 2, 3, 3, 40, 40])
    latent_bases = np.concatenate((latent_sizes[::-1].cumprod()[::-1][1:], np.array([1,])))
    return np.dot(latents, latent_bases).astype(int)

def findallfiles(BASE_PATH):
    for root, ds, fs in os.walk(BASE_PATH):
        for f in fs:
            fullname = os.path.join(root, f)
            yield fullname
            
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    train_loader, test_loader = get_dataloaders(args)
    for x,y,reg,idx in train_loader:
        break
    plt.imshow(x[0].transpose(1,2).transpose(0,2))
    """
    SOURCE_DIR = 'E:\\DATASET\\mpi3d_real\\real'
    TARGET_DIR = 'E:\\DATASET\\mpi3d_real\\select'
    g0 = 0
    index_all = []
    label_all = []
    for gg1 in range(6):
        g1 = gg1
        for gg2 in range(6):
            g2 = gg2
            g3 = 1
            g4 = 1
            g5 = 0
            for gg6 in range(10):
                g6 = gg6*4
                for gg7 in range(10):
                    g7 = gg7*4
                    tmp_idx = latent_to_index([g0,g1,g2,g3,g4,g5,g6,g7])
                    index_all.append(tmp_idx)
                    label_all.append(np.array([g1,g2,gg6,gg7]))
    index_all = np.array(index_all)
    label_all = np.array(label_all)
    
    image_all = []
    for idx in range(index_all.shape[0]):
    #for idx in range(2):    
        tgt_idx = index_all[idx]
        file_name = str(tgt_idx)+'.jpg'
        file_path = os.path.join(TARGET_DIR, file_name)
        img = Image.open(file_path)
        imgage_numpy = np.asarray(img)
        image_all.append(imgage_numpy)
    image_all = np.array(image_all)
    np.savez('E:\\DATASET\\mpi3d.npz', 
             images=image_all, 
             labels=label_all)

    #npzfile = np.load('E:\\DATASET\\mpi3d.npz')
    
    
    if False:
        for idx in range(index_all.shape[0]):
            print(idx)
            tgt_idx = index_all[idx]
            file_name = str(tgt_idx)+'.jpg'
            source_path = os.path.join(SOURCE_DIR, file_name)
            tgt_path = os.path.join(TARGET_DIR, file_name)
            shutil.copy2(source_path, tgt_path)
    """
    
        




