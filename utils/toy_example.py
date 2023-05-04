# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 13:20:38 2022

@author: YIREN
"""

import h5py
import numpy as np
import torch.utils.data as Data 
import torchvision.transforms as T
import os

PATH = '/home/joshua52/projects/def-dsuth/joshua52/P4_Graph/dataset/'
if not os.path.exists(PATH):
    PATH = 'E:\\DATASET\\'

class My_toy_Dataset(Data.Dataset):
    def __init__(self, x, y, reg, transform=None,):
        self.x = x
        self.y = y
        self.reg = reg
        self.transform=transform

    def __getitem__(self,index):
        img, y, reg, idx = self.x[index], self.y[index], self.reg[index], index
        if self.transform is not None:
            img = self.transform(img)       
        return img, y, reg, idx

    def __len__(self):
        return self.y.shape[0]

def get_reg_labels(oht_labels):
  PERM = np.array([
      [1,2,3,4],[1,3,2,4],[1,4,2,3],[2,3,1,4],[2,4,1,3],[3,4,1,2]
      ])-1
  reg_labels = []
  for i in range(PERM.shape[0]):
    AREA = np.random.randint(0,10,(4,1))
    #AREA = [1, 2, 0.5]
    id1,id2,id3,id4 = 0,1,2,3 #PERM[i]
    #reg_label = oht_labels[:,id1]/10*AREA[0] + oht_labels[:,id2]/10*AREA[1] + oht_labels[:,id3]*oht_labels[:,id4]/100*AREA[2]
    reg_label = oht_labels[:,id1]/10*AREA[0] + oht_labels[:,id2]/10*AREA[1] + oht_labels[:,id3]/10*AREA[2] + oht_labels[:,id4]/8*AREA[3]
    reg_label = (reg_label-reg_label.mean())/reg_label.std()
    reg_labels.append(reg_label)
  return np.array(reg_labels).transpose(1,0)

def generate_3dshape_loaders(args):
    #_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
    #_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 'scale': 8, 'shape': 4, 'orientation': 15}
    DATA_SIZE = 8000*args.data_per_g
    file_path = os.path.join(PATH, '3dshapes.h5')
    dataset = h5py.File(file_path, 'r')
    images = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
    labels = dataset['labels']  # array shape [480000,6], float64
     
    oht_labels = np.zeros((480000,4))
    oht_labels[:,0] = np.array(labels[:,0]*10,dtype=int)
    oht_labels[:,1] = np.array(labels[:,1]*10,dtype=int)
    oht_labels[:,2] = np.array(labels[:,2]*10,dtype=int)
    oht_labels[:,3] = np.array((labels[:,3]-0.75)*15,dtype=int)
    reg_labels = get_reg_labels(oht_labels)
    
    tmp = np.random.binomial(n=1,p=args.sup_ratio,size=(1,8000))
    mask_train = (tmp==1).squeeze()
    mask_test = (tmp==0).squeeze()
    tmp_img_idx = np.arange(0,8000,1)
    
    img_idx = tmp_img_idx*60+np.random.randint(0,60,size=(DATA_SIZE,))   # Random select other features
    #img_idx = tmp_img_idx*60+33   # Fix the other two features
    idx_train, idx_test  = img_idx[mask_train], img_idx[mask_test]
    input_train, input_test  = images[idx_train], images[idx_test] 
    label_train, label_test  = oht_labels[idx_train], oht_labels[idx_test] 
    reg_train, reg_test  = reg_labels[idx_train], reg_labels[idx_test]
    
    basic_T = T.Compose([T.ToTensor(), T.Resize([32,32])])
    dataset_train = My_toy_Dataset(input_train, label_train, reg_train, basic_T)
    dataset_test = My_toy_Dataset(input_test, label_test, reg_test, basic_T)
    train_loader = Data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last = True, num_workers=2)
    test_loader = Data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, drop_last = True, num_workers=2)
    
    return train_loader, test_loader

def generate_3dshape_fullloader_vae(args):
    #_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
    #_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 'scale': 8, 'shape': 4, 'orientation': 15}
    file_path = os.path.join(PATH, '3dshapes.h5')
    dataset = h5py.File(file_path, 'r')
    images = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
    labels = dataset['labels']  # array shape [480000,6], float64
     
    oht_labels = np.zeros((480000,4))
    oht_labels[:,0] = np.array(labels[:,0]*10,dtype=int)
    oht_labels[:,1] = np.array(labels[:,1]*10,dtype=int)
    oht_labels[:,2] = np.array(labels[:,2]*10,dtype=int)
    oht_labels[:,3] = np.array((labels[:,3]-0.75)*15,dtype=int)
    reg_labels = get_reg_labels(labels)
    
    tmp = np.random.binomial(n=1,p=args.sup_ratio,size=(1,480000))
    mask_sel = (tmp==1).squeeze()

    basic_T = T.Compose([T.ToTensor(), T.Resize([32,32])])
    input_all, label_all, reg_all = images[mask_sel], labels[mask_sel], reg_labels[mask_sel]
    dataset_all = My_toy_Dataset(input_all, label_all, reg_all, basic_T)
    all_loader = Data.DataLoader(dataset_all, batch_size=args.batch_size, shuffle=True, drop_last = True)

    return all_loader


# ============= dsprites ============

def latent_to_index(latents):
    latent_sizes = np.array([ 1,  3,  6, 40, 32, 32])
    latent_bases = np.concatenate((latent_sizes[::-1].cumprod()[::-1][1:], np.array([1,])))
    return np.dot(latents, latent_bases).astype(int)

def gen_train_test_indexes_dsprites(sup_ratio):
    g0 = 0
    index_all = []
    train_index, test_index = [], []
    for gg1 in range(3):
        g1 = gg1
        for gg2 in range(6):
            g2 = gg2
            #g3 = 5#np.random.randint(0,40,(1,))[0]
            for gg3 in range(5):
                g3 = gg3
                for gg4 in range(10):
                    g4 = gg4*3
                    for gg5 in range(10):
                        g5 = gg5*3
                        tmp_idx = latent_to_index([g0,g1,g2,g3,g4,g5])
                        # ----- Decide to go train or test
                        if np.random.binomial(n=1,p=sup_ratio,size=(1,)):
                            train_index.append(tmp_idx)
                        else:
                            test_index.append(tmp_idx)
    train_index = np.array(train_index)
    test_index = np.array(test_index)
    return train_index, test_index

def generate_dsprites_loaders(args):
    #_FACTORS_IN_ORDER = ['shape', 'scale', 'orientation', 'pos-x', 'pos-y']
    #_NUM_VALUES_PER_FACTOR = {'shape': 3, 'scale': 6, 'orientation': 40, 'x': 10 (32), 'y': 10 (32)}
    file_name = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    path = os.path.join(PATH, file_name)
    dataset = np.load(path,allow_pickle=True)
    images = dataset['imgs']
    values = dataset['latents_values']
    labels = dataset['latents_classes']
    tmp_value = np.delete(values,[0,3],axis=1)
    regs = get_reg_labels(tmp_value)
    
    idx_train, idx_test = gen_train_test_indexes_dsprites(args.sup_ratio)
    train_y, test_y = values[idx_train], values[idx_test]
    train_y, test_y = np.delete(train_y,[0,3],axis=1), np.delete(test_y,[0,3],axis=1)
    train_cls, test_cls = labels[idx_train], labels[idx_test]
    train_cls[:,-2:], test_cls[:,-2:] = train_cls[:,-2:]/3, test_cls[:,-2:]/3
    
    input_train, input_test = images[idx_train], images[idx_test]
    reg_train, reg_test = regs[idx_train], regs[idx_test]
    label_train, label_test = np.delete(train_cls,[0,3],axis=1), np.delete(test_cls,[0,3],axis=1)
    
    basic_T = T.Compose([T.ToTensor()])
    dataset_train = My_toy_Dataset(input_train, label_train, reg_train, basic_T)
    dataset_test = My_toy_Dataset(input_test, label_test, reg_test, basic_T)
    train_loader = Data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last = True, num_workers=2)
    test_loader = Data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last = True, num_workers=2)   

    return train_loader, test_loader

def generate_dsprites_fullloader_vae(args):
    #_FACTORS_IN_ORDER = ['shape', 'scale', 'orientation', 'pos-x', 'pos-y']
    #_NUM_VALUES_PER_FACTOR = {'shape': 3, 'scale': 6, 'orientation': 40, 'x': 10 (32), 'y': 10 (32)}
    file_name = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    path = os.path.join(PATH, file_name)
    dataset = np.load(path,allow_pickle=True)
    
    tmp = np.random.binomial(n=1,p=args.sup_ratio,size=(1,737280))
    mask_sel = (tmp==1).squeeze()
    
    images = dataset['imgs'][mask_sel]
    values = dataset['latents_values'][mask_sel]
    labels = dataset['latents_classes'][mask_sel]
    basic_T = T.Compose([T.ToTensor()])
    dataset_all = My_toy_Dataset(images, labels, values,basic_T)
    full_loader = Data.DataLoader(dataset_all, batch_size=args.batch_size, shuffle=True, drop_last = True, num_workers=2)
    return full_loader

#================ MPI3D ================
def generate_mpi3d_loaders(args):
    #_FACTORS_IN_ORDER = ['shape', 'scale', 'orientation', 'pos-x', 'pos-y']
    #_NUM_VALUES_PER_FACTOR = {'shape': 3, 'scale': 6, 'orientation': 40, 'x': 10 (32), 'y': 10 (32)}
    file_name = "mpi3d.npz"
    path = os.path.join(PATH, file_name)
    dataset = np.load(path,allow_pickle=True)
    images = dataset['images']
    labels = dataset['labels']   
    regs = get_reg_labels(labels)
    
    tmp = np.random.binomial(n=1,p=args.sup_ratio,size=(1,3600))
    mask_train = (tmp==1).squeeze()
    mask_test = (tmp==0).squeeze()
    basic_T = T.Compose([T.ToTensor(), T.Resize([32,32])])
    
    input_train, input_test = images[mask_train], images[mask_test]
    label_train, label_test = labels[mask_train], labels[mask_test]
    reg_train, reg_test = regs[mask_train], regs[mask_test]
    
    dataset_train = My_toy_Dataset(input_train, label_train, reg_train,basic_T)
    dataset_test = My_toy_Dataset(input_test, label_test, reg_test,basic_T)
    train_loader = Data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last = True, num_workers=2)
    test_loader = Data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last = True, num_workers=2)   

    return train_loader, test_loader

def generate_mpi3d_fullloader_vae(args):
    pass

def get_dataloaders(args):
    if args.dataset_name=='3dshapes':
        a, b = generate_3dshape_loaders(args)
        return a, b
    elif args.dataset_name=='dsprites':
        a, b = generate_dsprites_loaders(args)
        return a, b
    elif args.dataset_name=='mpi3d':
        a, b = generate_mpi3d_loaders(args)
        return a, b
    
def get_vae_loader(args):
    if args.dataset_name=='3dshapes':
        return generate_3dshape_fullloader_vae(args)
    elif args.dataset_name=='dsprites':
        return generate_dsprites_fullloader_vae(args)
    elif args.dataset_name=='mpi3d':
        return generate_mpi3d_fullloader_vae(args)    

'''
def generate_small_3dshape_loaders(args):
    #_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
    #_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 'scale': 8, 'shape': 4, 'orientation': 15}
    DATA_SIZE = 8000*args.data_per_g
    #dataset = h5py.File('/home/mila/y/yi.ren/P4_Graph/dataset/3dshapes.h5', 'r')
    dataset = h5py.File('E:\\DATASET\\3dshapes.h5', 'r')
    images = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
    labels = dataset['labels']  # array shape [480000,6], float64
    image_shape = images.shape[1:]  # [64,64,3]
    label_shape = labels.shape[1:]  # [6]
    n_samples = labels.shape[0]  # 10*10*10*8*4*15=480000
    
    AREA = [1, 2, 0.5]
    
    oht_labels = np.zeros((100,4))
    oht_labels[:,0] = np.array(labels[:100,0]*10,dtype=int)
    oht_labels[:,1] = np.array(labels[:100,1]*10,dtype=int)
    oht_labels[:,2] = np.array(labels[:100,2]*10,dtype=int)
    oht_labels[:,3] = np.array((labels[:,3]-0.75)*15,dtype=int)
    reg_labels = oht_labels[:,0]/10*AREA[0] + oht_labels[:,1]/10*AREA[1] + oht_labels[:,2]*oht_labels[:,3]/80*AREA[2]
    #reg_labels = (reg_labels-reg_labels.mean())/reg_labels.std()
    
    tmp = np.random.binomial(n=1,p=args.sup_ratio,size=(1,100))
    mask_train = (tmp==1).squeeze()
    mask_test = (tmp==0).squeeze()
    tmp_img_idx = np.arange(0,100,1)
    
    img_idx = tmp_img_idx*1
    idx_train, idx_test  = img_idx[mask_train], img_idx[mask_test]
    input_train, input_test  = images[idx_train], images[idx_test] 
    label_train, label_test  = oht_labels[idx_train], oht_labels[idx_test] 
    reg_train, reg_test  = reg_labels[idx_train], reg_labels[idx_test]
    
    # ====== Make extra unsupervised samples if nessary
    idx_unsup = tmp_img_idx*1
    input_unsup, label_unsup, reg_unsup = images[idx_unsup], oht_labels[idx_unsup], reg_labels[idx_unsup]

    basic_T = T.Compose([T.ToTensor(), T.Resize([32,32])])
    dataset_train = My_toy_Dataset(input_train, label_train, reg_train, basic_T)
    dataset_test = My_toy_Dataset(input_test, label_test, reg_test, basic_T)
    dataset_unsup = My_toy_Dataset(input_unsup, label_unsup, reg_unsup, basic_T)
    train_loader = Data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last = True, num_workers=2)
    test_loader = Data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, drop_last = True, num_workers=2)
    unsup_loader = Data.DataLoader(dataset_unsup, batch_size=args.batch_size, shuffle=True, drop_last = True, num_workers=2)
    
    return train_loader, test_loader, unsup_loader
'''
    