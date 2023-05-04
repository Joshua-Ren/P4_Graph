# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 21:39:12 2022

@author: YIREN
"""
import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn.init as init

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=38):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_classes=num_classes
        self.expansion = block.expansion
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.Bob = nn.Sequential(
                nn.Linear(512*block.expansion, self.num_classes)
                )
        #self.linear = nn.Linear(512*block.expansion, num_classes)
        #self.linear2 = nn.Linear(128,num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        hid = out.view(out.size(0), -1)
        pred = self.Bob(hid)
        #out = self.linear2(out)
        return hid, pred

class ResNet_SEM(ResNet):
    def __init__(self, L=4, V=10, tau=1., **kwargs):
        super(ResNet_SEM, self).__init__(**kwargs)
        # ------ SEM Part
        self.L = L
        self.V = V
        self.tau = tau
        self.Wup = nn.Linear(512*self.expansion, self.L*self.V)    #Split the linear by Wup and Whead
        #self.linear = nn.Linear(self.L*self.V, num_classes)
        self.Bob = nn.Sequential(
                nn.Linear(self.L*self.V, self.num_classes)
                )
        #self.linear = nn.Sequential(
        #                nn.Linear(self.L*self.V, 128),
        #                nn.ReLU(),
        #                nn.Linear(128, num_classes)
        #)

    def SEM(self, in_vector):
        '''
            Piecewise softmax on a long 1*(L*V) vector
            Use tau to control the softmax temperature
            e.g., embd_size=300, L=30, V=10, as we have 30 words, each with 10 possible choices
        '''
        b_size = in_vector.shape[0]
        w_invector = self.Wup(in_vector)    # N*512 --> N*40
        w_invector = w_invector/self.tau
        msg = w_invector.reshape(b_size, self.L, self.V)
        p_theta = torch.nn.Softmax(-1)(msg).reshape(b_size, -1)   # reshaped prob-logits
        return msg, p_theta

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        hid = out.view(out.size(0), -1)
        msg, sem_hid = self.SEM(hid) 
        out = self.Bob(sem_hid)
        return msg, out

class ResNet_VAE(ResNet):
    def __init__(self, z_dim=10, **kwargs):
        super(ResNet_VAE, self).__init__(**kwargs)
        # ------ VAE part
        self.z_dim = z_dim
        self.linear_toz = nn.Linear(512*self.expansion, self.z_dim*2)
        self.linear_toL = nn.Sequential(
                nn.Linear(self.z_dim, self.num_classes)
                )

    def ds_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        z = self.linear_toz(out)
        out2 = self.linear_toL(z[:,:self.z_dim])
        return z, out2

    def forward(self, x):
        # ----- 
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        hid = out.view(out.size(0), -1)
        out2 = self.linear_toz(hid)
        return out2

class MLP_ML(nn.Module):
  def __init__(self, L=4, V=10, in_dim=3072, hid_size=128, num_classes=38):
    super(MLP_ML, self).__init__()
    self.in_dim = in_dim
    self.L = L
    self.V = V
    self.num_classes = num_classes
    self.hid_size = hid_size
    self.Alice = nn.Sequential(
              nn.Linear(self.in_dim, self.hid_size),
              nn.ReLU(True),
              nn.Linear(self.hid_size, self.hid_size),
              nn.ReLU(True),
              nn.Linear(self.hid_size, self.hid_size),
              nn.ReLU(True),
            )
    self.Wup = nn.Linear(self.hid_size, self.L*self.V)
    self.Bob = nn.Sequential(
              nn.Linear(self.L*self.V, self.num_classes)
            )
  def forward(self, x):
    x = x.view(x.size(0),-1)
    z = self.Alice(x)
    z = self.Wup(z)
    out = self.Bob(z)
    return z, out

class MLP_SEM(nn.Module):
  def __init__(self, L=4, V=10, tau=1., in_dim=3072, hid_size=128, num_classes=38):
    super(MLP_SEM, self).__init__()
    self.in_dim = in_dim
    self.num_classes = num_classes
    self.hid_size = hid_size
    self.Alice = nn.Sequential(
              nn.Linear(self.in_dim, self.hid_size),
              nn.ReLU(True),
              nn.Linear(self.hid_size, self.hid_size),
              nn.ReLU(True),
              nn.Linear(self.hid_size, self.hid_size),
              nn.ReLU(True),
            )
    self.L = L
    self.V = V
    self.tau = tau
    self.Wup = nn.Linear(self.hid_size, self.L*self.V)    #Split the linear by Wup and Whead

    self.Bob = nn.Sequential(
              nn.Linear(self.L*self.V, self.num_classes)
            )
  def SEM(self, in_vector):
      '''
          Piecewise softmax on a long 1*(L*V) vector
          Use tau to control the softmax temperature
          e.g., embd_size=300, L=30, V=10, as we have 30 words, each with 10 possible choices
      '''
      b_size = in_vector.shape[0]
      w_invector = self.Wup(in_vector)    # N*512 --> N*40
      w_invector = w_invector/self.tau
      msg = w_invector.reshape(b_size, self.L, self.V)
      p_theta = torch.nn.Softmax(-1)(msg).reshape(b_size, -1)   # reshaped prob-logits
      return msg, p_theta

  def forward(self, x):
    x = x.view(x.size(0),-1)
    z = self.Alice(x)
    msg, sem_hid = self.SEM(z)
    out = self.Bob(sem_hid)
    return msg, out

class CNN_SEM(nn.Module):
  def __init__(self, L=4, V=10, tau=1., hid_size=128, num_classes=6, sem_flag=True):
    super(CNN_SEM, self).__init__()
    self.sem_flag = sem_flag
    self.num_classes = num_classes
    self.hid_size = hid_size
    self.Alice = nn.Sequential(
              nn.Conv2d(1, 32, 4, 2, 1),          # B,  32, 32, 32
              nn.ReLU(True),
              nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
              nn.ReLU(True),
              nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
              nn.ReLU(True),
              nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
              nn.ReLU(True),
              nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
              nn.ReLU(True),
              View((-1, 256*1*1)),                 # B, 256
              nn.Linear(256, self.hid_size),             # B, z_dim*
            )
    self.L = L
    self.V = V
    self.tau = tau
    self.Wup = nn.Sequential(
              nn.Linear(self.hid_size, self.L*self.V)    #Split the linear by Wup and Whead
            )

    self.Bob = nn.Sequential(
              nn.Linear(self.L*self.V, self.num_classes)
            )
    self.weight_init()

  def weight_init(self):
      for block in self._modules:
          for m in self._modules[block]:
              kaiming_init(m)
  def SEM(self, in_vector):
      '''
          Piecewise softmax on a long 1*(L*V) vector
          Use tau to control the softmax temperature
          e.g., embd_size=300, L=30, V=10, as we have 30 words, each with 10 possible choices
      '''
      b_size = in_vector.shape[0]
      w_invector = self.Wup(in_vector)    # N*512 --> N*40
      w_invector = w_invector/self.tau
      msg = w_invector.reshape(b_size, self.L, self.V)
      if self.sem_flag:
        p_theta = torch.nn.Softmax(-1)(msg).reshape(b_size, -1)   # reshaped prob-logits
      else:
        p_theta = msg.reshape(b_size, -1)
      return msg, p_theta

  def forward(self, x):
    x = x.unsqueeze(1)
    z = self.Alice(x)
    msg, sem_hid = self.SEM(z)
    out = self.Bob(sem_hid)
    return msg, out

def ResNet18_ML(num_classes=1):
    return ResNet(block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=num_classes)

def ResNet18_SEM(L=4, V=10, tau=1., num_classes=1):
    return ResNet_SEM(block=BasicBlock, num_blocks=[2, 2, 2, 2],L=L,V=V,tau=tau,num_classes=num_classes)

def ResNet18_VAE(z_dim=10, num_classes=1):
    return ResNet_VAE(block=BasicBlock, num_blocks=[2, 2, 2, 2], z_dim=z_dim, num_classes=num_classes)