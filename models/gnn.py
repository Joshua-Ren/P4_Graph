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
        self.task_head = nn.Sequential(
                            nn.Linear(self.emb_dim, self.num_tasks),
                            #nn.ReLU(),
                            #nn.Linear(self.emb_dim, self.num_tasks),
                            )        
          
    def task_forward(self, batched_data, sem_tau=1.):
        # downstream task forward
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        output = self.task_head(h_graph)
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
    def __init__(self, L=200, V=20, tau=1., **kwargs):
        super(GNN_SEM_UPSAMPLE, self).__init__(**kwargs)
        self.L = L
        self.V = V
        self.Wup = nn.Linear(self.emb_dim, self.L*self.V)
        #self.Wq = nn.Linear(self.L*self.V, self.L*self.V)
        #self.task_head = nn.Sequential(
                            #nn.Linear(self.L*self.V, self.num_tasks),
        #                    nn.Linear(self.L*self.V, self.emb_dim, bias=False),
        #                    nn.Linear(self.emb_dim, self.num_tasks)
        #                    )
        self.Vocab_embd = nn.Linear(self.V, self.emb_dim,bias=False)
        self.Vocab_task = nn.Linear(self.L, self.num_tasks,bias=False)
        self.Combi_L = nn.Linear(self.emb_dim, 1)

    def task_head(self, in_vector):
        b_size = in_vector.shape[0]
        in_vector = in_vector.reshape(b_size, self.L, self.V)
        h1 = self.Vocab_embd(in_vector)  # h1 shape: B, L, emb_dim
        h1 = h1.transpose(1,2)           # h1 shape: B, emb_dim, L
        h2 = self.Vocab_task(h1)         # h2 shape: B, emb_dim, num_task
        h2 = h2.transpose(1,2)           # h2 shape: B, num_task, emb_dim
        h3 = self.Combi_L(h2)            # h3 shape: B, num_task, 1
        return h3.squeeze(-1)
        
    def SEM(self, in_vector, tau=1.):
        '''
            Piecewise softmax on a long 1*(L*V) vector
            Use tau to control the softmax temperature
            e.g., embd_size=300, L=30, V=10, as we have 30 words, each with 10 possible choices
        '''
        b_size = in_vector.shape[0]
        w_invector = self.Wup(in_vector)    # N*300 --> N*4000
        logits = w_invector.reshape(b_size, self.L, self.V)
        w_invector = w_invector/tau
        logits_tau = w_invector.reshape(b_size, self.L, self.V)
        p_theta = torch.nn.Softmax(-1)(logits_tau).reshape(b_size, -1)   # reshaped prob-logits
        #q_theta = self.Wq(p_theta)  # q only for BYOL
        return logits, p_theta
          
    def task_forward(self, batched_data, sem_tau=1.):
        # downstream task forward
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        logits, p_theta = self.SEM(h_graph, sem_tau)    
        output = self.task_head(p_theta)
        return logits, output

    def distill_forward(self, batched_data, sem_tau=1.):
        # for distill, both use logits
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch) 
        logits, p_theta = self.SEM(h_graph, sem_tau)
        return logits, p_theta   

if __name__ == '__main__':
    GNN(num_tasks = 10)