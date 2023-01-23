# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 20:35:45 2022

@author: YIREN
"""

# possible datasets: (size,batch.num_nodes,y.shape)
# molpcba(10949-763-128), molhiv(1029-852-1), molbace (38-1103-1), 
# molbbbp(51-781-1), molclintox(37-858-2), molmuv(2328-764-17), molsider(36-1502-27), moltox21(196-431-12), 
# moltoxcast(215-455-617), molesol(29-417-1), molfreesolv(17-257-1), mollipo(105-783-1)

import torch
import torch.nn as nn
from torch_geometric import nn as gnn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform
from models.conv import GNN_node, GNN_node_Virtualnode
from models.lstm_sem import MsgGenLSTM, MsgDecoderLSTM
from torch_scatter import scatter_mean

class GNN(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            #self.graph_pred_linear = nn.Linear(2*self.emb_dim, self.num_tasks)
            self.graph_pred_linear = nn.Sequential(
                                            nn.Linear(2*self.emb_dim, self.num_tasks),
                                            )
                                            
        else:
            #self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)
            self.graph_pred_linear = nn.Sequential(
                                            nn.Linear(self.emb_dim, self.num_tasks),
                                            )
    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        return self.graph_pred_linear(h_graph)
        
    def dis_foward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)   
        return h_graph

class GNN_STD(GNN):
    '''
        Standard GNN, no SEM or bottleneck
    '''
    def __init__(self, L=200, V=20, tau=1., **kwargs):
        super(GNN_STD, self).__init__(**kwargs)
        self.linear_head = nn.Sequential(
                            nn.Linear(self.emb_dim, self.num_tasks),
                            #nn.ReLU(),
                            #nn.Linear(self.emb_dim, self.num_tasks),
                            )        
          
    def task_forward(self, batched_data, sem_tau=1.):
        # downstream task forward
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        output = self.linear_head(h_graph)
        return h_graph, output

    def distill_forward(self, batched_data, sem_tau=1.):
        # for distill, both use logits
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        return h_graph, h_graph

class GNN_SEM_UPSAMPLE(GNN):
    '''
        For design 2, the message matrix has shape L*V
        Task head is L*V->L, then L->Ntask
        For distillation, output long L*V vector
        For SSL, need more Wq and Wg
    '''
    def __init__(self, L=200, V=20, tau=1., head_type='linear', **kwargs):
        super(GNN_SEM_UPSAMPLE, self).__init__(**kwargs)
        self.L = L
        self.V = V
        self.head_type = head_type
        self.Wup = nn.Linear(self.emb_dim, self.L*self.V)
        self.BN = nn.BatchNorm1d(self.L*self.V)
        self.linear_head = nn.Sequential(
                            nn.Linear(self.L*self.V, self.num_tasks),
                            )
        self.mlp_head = nn.Sequential(
                            nn.Linear(self.L*self.V, self.emb_dim),
                            nn.ReLU(),
                            nn.Linear(self.emb_dim, self.num_tasks)
                            )       
    def SEM(self, in_vector, tau=1.):
        '''
            Piecewise softmax on a long 1*(L*V) vector
            Use tau to control the softmax temperature
            e.g., embd_size=300, L=30, V=10, as we have 30 words, each with 10 possible choices
        '''
        b_size = in_vector.shape[0]
        w_invector = self.Wup(in_vector)    # N*300 --> N*4000
        logits = w_invector.reshape(b_size, self.L, self.V)
        
        w_invector = self.BN(w_invector)
        
        w_invector = w_invector/tau
        logits_tau = w_invector.reshape(b_size, self.L, self.V)
        z_hat = torch.nn.Softmax(-1)(logits_tau).reshape(b_size, -1)   # reshaped prob-logits
        #q_theta = self.Wq(z_hat)  # q only for BYOL
        return logits, z_hat
          
    def task_forward(self, batched_data, sem_tau=1.):
        # downstream task forward
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        logits, z_hat = self.SEM(h_graph, sem_tau)
        if self.head_type=='mlp':
            output = self.mlp_head(z_hat)
        else:
            output = self.linear_head(z_hat)
        return logits, output

    def distill_forward(self, batched_data, sem_tau=1.):
        # for distill, both use logits
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch) 
        logits, z_hat = self.SEM(h_graph, sem_tau)
        return logits, z_hat   


class GNN_SEM_BASELINE(GNN):
    '''
        For design 2, the message matrix has shape L*V
        Task head is L*V->L, then L->Ntask
        For distillation, output long L*V vector
        For SSL, need more Wq and Wg
    '''
    def __init__(self, L=200, V=20, tau=1., head_type='linear', **kwargs):
        super(GNN_SEM_BASELINE, self).__init__(**kwargs)
        self.L = L
        self.V = V
        self.head_type = head_type
        self.Wup = nn.Linear(self.emb_dim, self.L*self.V)
        self.BN = nn.BatchNorm1d(self.L*self.V)
        self.linear_head = nn.Sequential(
                            nn.Linear(self.L*self.V, self.num_tasks),
                            )
        self.mlp_head = nn.Sequential(
                            nn.Linear(self.L*self.V, self.emb_dim),
                            nn.ReLU(),
                            nn.Linear(self.emb_dim, self.num_tasks)
                            )       
    def SEM(self, in_vector, tau=1.):
        '''
            Piecewise softmax on a long 1*(L*V) vector
            Use tau to control the softmax temperature
            e.g., embd_size=300, L=30, V=10, as we have 30 words, each with 10 possible choices
        '''
        b_size = in_vector.shape[0]
        w_invector = self.Wup(in_vector)    # N*300 --> N*4000
        logits = w_invector.reshape(b_size, self.L, self.V)
        
        w_invector = self.BN(w_invector)
        
        w_invector = w_invector/tau
        logits_tau = w_invector.reshape(b_size, self.L, self.V)
        z_hat = logits_tau.reshape(b_size, -1)   # reshaped prob-logits
        #q_theta = self.Wq(z_hat)  # q only for BYOL
        return logits, z_hat
          
    def task_forward(self, batched_data, sem_tau=1.):
        # downstream task forward
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        logits, z_hat = self.SEM(h_graph, sem_tau)
        if self.head_type=='mlp':
            output = self.mlp_head(z_hat)
        else:
            output = self.linear_head(z_hat)
        return logits, output

    def distill_forward(self, batched_data, sem_tau=1.):
        # for distill, both use logits
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch) 
        logits, z_hat = self.SEM(h_graph, sem_tau)
        return logits, z_hat

if __name__ == '__main__':
    GNN(num_tasks = 10)