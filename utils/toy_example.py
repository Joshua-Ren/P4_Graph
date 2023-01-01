# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 13:20:38 2022

@author: YIREN
"""

import h5py
import numpy as np
import torch.utils.data as Data 
import torchvision.transforms as T

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

def generate_3dshape_loaders(args):
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
    
    oht_labels = np.zeros((480000,4))
    oht_labels[:,0] = np.array(labels[:,0]*10,dtype=int)
    oht_labels[:,1] = np.array(labels[:,1]*10,dtype=int)
    oht_labels[:,2] = np.array(labels[:,2]*10,dtype=int)
    oht_labels[:,3] = np.array(labels[:,0]*8,dtype=int)
    reg_labels = oht_labels[:,0]/10*AREA[0] + oht_labels[:,1]/10*AREA[1] + oht_labels[:,2]*oht_labels[:,3]/80*AREA[2]
    reg_labels = (reg_labels-reg_labels.mean())/reg_labels.std()
    
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
    
    # ====== Make extra unsupervised samples if nessary
    idx_unsup = tmp_img_idx*60+np.random.randint(0,60,size=(tmp_img_idx.shape[0],))   # Random select other features
    idx_unsup = np.unique(np.sort(np.random.randint(0,480000,size=(8000,))))
    #idx_unsup = get_unsup_samples(ratio=5000)
    input_unsup, label_unsup, reg_unsup = images[idx_unsup], oht_labels[idx_unsup], reg_labels[idx_unsup]

    basic_T = T.Compose([T.ToTensor(), T.Resize([32,32])])
    dataset_train = My_toy_Dataset(input_train, label_train, reg_train, basic_T)
    dataset_test = My_toy_Dataset(input_test, label_test, reg_test, basic_T)
    dataset_unsup = My_toy_Dataset(input_unsup, label_unsup, reg_unsup, basic_T)
    train_loader = Data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last = True, num_workers=2)
    test_loader = Data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True, drop_last = True, num_workers=2)
    unsup_loader = Data.DataLoader(dataset_unsup, batch_size=args.batch_size, shuffle=True, drop_last = True, num_workers=2)
    
    return train_loader, test_loader, unsup_loader


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
    oht_labels[:,3] = np.array(labels[:100,0]*8,dtype=int)
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
    