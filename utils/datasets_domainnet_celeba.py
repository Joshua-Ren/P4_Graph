# -*- coding: utf-8 -*-
"""
y: 0-87, sketch
   88-134, real
   135-199, quick
"""

import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import os
import pandas as pd
import numpy as np
import random

DATA_PATH = "E:\\DATASET"
#DATA_PATH = "/home/joshua52/scratch/dataset"
traindir = os.path.join(DATA_PATH, 'train')
valdir = os.path.join(DATA_PATH, 'test')

def get_std_transform(figsize=256, cropsize=224):
    """
        For CIFAR10/100, STL, Domain Net or other small dataset, use this
    """
    train_T=T.Compose([
                    T.Resize([figsize,figsize]),
                    T.RandomCrop(cropsize),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
    val_T =T.Compose([
                    T.Resize([cropsize,cropsize]),
                    T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                    ])
    return train_T, val_T


def generate_celeba_loader(args):
    train_T, val_T = get_std_transform()
    train_dataset = datasets.CelebA(DATA_PATH, split='train', transform=train_T, download=False)
    val_dataset = datasets.CelebA(DATA_PATH, split='valid', transform=val_T, download=False)
    #train_dataset = val_dataset
    #indices = torch.randperm(len(train_dataset))[:int(len(train_dataset) * 0.5)]
    indices = torch.tensor(np.arange(0,len(train_dataset), 4))   # Downsample the dataset
    train_dataset = Data.Subset(train_dataset, indices)

    train_loader = Data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    val_loader = Data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=2)
    return train_loader, val_loader

"""
rand_index = list(np.arange(0,200,1))
random.shuffle(rand_index)
LABEL_MAPPINGS = {}
for i in range(200):
    G1 = rand_index[i]
    if i<=87:
        G2 = 0
    elif i<=134:
        G2 = 1
    else:
        G2 = 2
    LABEL_MAPPINGS[i] = []
    LABEL_MAPPINGS[i].append(G1)
    LABEL_MAPPINGS[i].append(G2)
"""

def generate_domainnet_loader(args): 
    train_T, val_T = get_std_transform(figsize=224)
    train_dataset = datasets.ImageFolder(traindir, transform=train_T)
    valid_dataset = datasets.ImageFolder(valdir, transform=val_T)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=args.batch_size, pin_memory=True, drop_last=True)
    return train_loader, valid_loader

def label_mapping(y):
    bsize = y.shape[0]
    y1 = torch.zeros_like(y)
    y2 = torch.zeros_like(y)
    for i in range(bsize):
        idx = y[i].item()
        y1[i] = LABEL_MAPPINGS[idx][0]
        y2[i] = LABEL_MAPPINGS[idx][1]
    return y1, y2


LABEL_MAPPINGS = {0: [185, 0],
 1: [9, 0],
 2: [136, 0],
 3: [179, 0],
 4: [87, 0],
 5: [36, 0],
 6: [196, 0],
 7: [137, 0],
 8: [105, 0],
 9: [62, 0],
 10: [40, 0],
 11: [189, 0],
 12: [181, 0],
 13: [89, 0],
 14: [110, 0],
 15: [57, 0],
 16: [65, 0],
 17: [30, 0],
 18: [1, 0],
 19: [155, 0],
 20: [175, 0],
 21: [53, 0],
 22: [159, 0],
 23: [193, 0],
 24: [71, 0],
 25: [48, 0],
 26: [134, 0],
 27: [172, 0],
 28: [90, 0],
 29: [117, 0],
 30: [161, 0],
 31: [39, 0],
 32: [15, 0],
 33: [66, 0],
 34: [58, 0],
 35: [198, 0],
 36: [50, 0],
 37: [3, 0],
 38: [60, 0],
 39: [128, 0],
 40: [81, 0],
 41: [146, 0],
 42: [156, 0],
 43: [0, 0],
 44: [123, 0],
 45: [113, 0],
 46: [163, 0],
 47: [19, 0],
 48: [147, 0],
 49: [73, 0],
 50: [70, 0],
 51: [118, 0],
 52: [115, 0],
 53: [157, 0],
 54: [86, 0],
 55: [31, 0],
 56: [12, 0],
 57: [111, 0],
 58: [88, 0],
 59: [96, 0],
 60: [69, 0],
 61: [28, 0],
 62: [33, 0],
 63: [188, 0],
 64: [194, 0],
 65: [21, 0],
 66: [16, 0],
 67: [49, 0],
 68: [68, 0],
 69: [45, 0],
 70: [116, 0],
 71: [195, 0],
 72: [55, 0],
 73: [77, 0],
 74: [144, 0],
 75: [92, 0],
 76: [125, 0],
 77: [34, 0],
 78: [25, 0],
 79: [93, 0],
 80: [75, 0],
 81: [61, 0],
 82: [171, 0],
 83: [109, 0],
 84: [95, 0],
 85: [52, 0],
 86: [46, 0],
 87: [173, 0],
 88: [35, 1],
 89: [131, 1],
 90: [20, 1],
 91: [78, 1],
 92: [14, 1],
 93: [140, 1],
 94: [162, 1],
 95: [8, 1],
 96: [74, 1],
 97: [182, 1],
 98: [129, 1],
 99: [59, 1],
 100: [141, 1],
 101: [160, 1],
 102: [127, 1],
 103: [6, 1],
 104: [91, 1],
 105: [121, 1],
 106: [180, 1],
 107: [152, 1],
 108: [133, 1],
 109: [190, 1],
 110: [168, 1],
 111: [199, 1],
 112: [103, 1],
 113: [158, 1],
 114: [76, 1],
 115: [120, 1],
 116: [167, 1],
 117: [82, 1],
 118: [104, 1],
 119: [126, 1],
 120: [29, 1],
 121: [130, 1],
 122: [145, 1],
 123: [176, 1],
 124: [94, 1],
 125: [27, 1],
 126: [191, 1],
 127: [7, 1],
 128: [153, 1],
 129: [148, 1],
 130: [166, 1],
 131: [101, 1],
 132: [98, 1],
 133: [44, 1],
 134: [187, 1],
 135: [2, 2],
 136: [37, 2],
 137: [56, 2],
 138: [18, 2],
 139: [32, 2],
 140: [107, 2],
 141: [170, 2],
 142: [83, 2],
 143: [139, 2],
 144: [184, 2],
 145: [23, 2],
 146: [26, 2],
 147: [54, 2],
 148: [183, 2],
 149: [85, 2],
 150: [164, 2],
 151: [178, 2],
 152: [17, 2],
 153: [119, 2],
 154: [38, 2],
 155: [132, 2],
 156: [63, 2],
 157: [79, 2],
 158: [4, 2],
 159: [138, 2],
 160: [124, 2],
 161: [43, 2],
 162: [135, 2],
 163: [112, 2],
 164: [114, 2],
 165: [174, 2],
 166: [97, 2],
 167: [72, 2],
 168: [41, 2],
 169: [24, 2],
 170: [177, 2],
 171: [108, 2],
 172: [142, 2],
 173: [84, 2],
 174: [154, 2],
 175: [99, 2],
 176: [122, 2],
 177: [67, 2],
 178: [149, 2],
 179: [64, 2],
 180: [10, 2],
 181: [197, 2],
 182: [5, 2],
 183: [22, 2],
 184: [42, 2],
 185: [47, 2],
 186: [13, 2],
 187: [151, 2],
 188: [192, 2],
 189: [80, 2],
 190: [51, 2],
 191: [106, 2],
 192: [150, 2],
 193: [102, 2],
 194: [186, 2],
 195: [11, 2],
 196: [165, 2],
 197: [169, 2],
 198: [143, 2],
 199: [100, 2]}















