# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 22:15:43 2023

@author: YIREN
"""
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNet_SEM_ML(nn.Module):
    def __init__(self, L=4, V=10, num_classes=40, tau=1., sem_flag=True, pretrain_flag=True):
        super(ResNet_SEM_ML, self).__init__()
        # ------ SEM Part
        self.L = L
        self.V = V
        self.tau = tau
        self.sem_flag = sem_flag
        self.Wup = nn.Linear(512, self.L*self.V, bias=False)    #Split the linear by Wup and Whead        
        self.Bob = nn.Sequential(
                    nn.Linear(self.L*self.V, 256),
                    nn.ReLU(),
                    nn.Linear(256, num_classes)
                    )
        self.model = models.resnet18(pretrained=pretrain_flag)
        self.model.fc = self.Wup
        
    def SEM(self, in_vector):
        '''
            Piecewise softmax on a long 1*(L*V) vector
            Use tau to control the softmax temperature
            e.g., embd_size=300, L=30, V=10, as we have 30 words, each with 10 possible choices
        '''
        b_size = in_vector.shape[0]
        #w_invector = self.Wup(in_vector)    # N*512 --> N*40, directly change model.fc in __init__
        w_invector = in_vector    
        w_invector = w_invector/self.tau
        msg = w_invector.reshape(b_size, self.L, self.V)
        if self.sem_flag:
            p_theta = nn.Softmax(-1)(msg).reshape(b_size, -1)   # reshaped prob-logits
        else:
            p_theta = msg.reshape(b_size, -1)
        return msg, p_theta
    
    def forward(self, x):
        hid = self.model(x)
        msg, sem_hid = self.SEM(hid)
        out = self.Bob(sem_hid)
        return msg, out
        
if __name__ == '__main__':
    model = ResNet_SEM_ML(L=10, V=40, tau=1., num_classes=40, sem_flag=True)



