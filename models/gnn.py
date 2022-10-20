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
                            nn.Linear(self.emb_dim, self.emb_dim),
                            nn.ReLU(),
                            nn.Linear(self.emb_dim, self.num_tasks),
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

class GNN_STD_ATT(GNN):
    '''
        Standard GNN, no SEM or bottleneck
    '''
    def __init__(self, L=200, V=20, tau=1., **kwargs):
        super(GNN_STD_ATT, self).__init__(**kwargs)
        self.task_head = nn.Sequential(
                            nn.Linear(self.emb_dim, self.emb_dim),
                            nn.ReLU(),
                            nn.Linear(self.emb_dim, self.num_tasks),
                            )      
        self.pool = GlobalAttention(gate_nn = 
                        nn.Sequential(torch.nn.Linear(self.emb_dim, 2*self.emb_dim), 
                        nn.BatchNorm1d(2*self.emb_dim),
                        nn.ReLU(),
                        nn.Linear(2*self.emb_dim, 1)))
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
        self.Wq = nn.Linear(self.L*self.V, self.L*self.V)
        self.task_head = nn.Sequential(
                            nn.Linear(self.L*self.V, self.num_tasks),
                            #nn.Linear(self.L*self.V, self.L, bias=False),
                            #nn.ReLU(),
                            #nn.Linear(self.L, self.num_tasks)
                            )
        
    def SEM(self, in_vector, tau=1.):
        '''
            Piecewise softmax on a long 1*(L*V) vector
            Use tau to control the softmax temperature
            e.g., embd_size=300, L=30, V=10, as we have 30 words, each with 10 possible choices
        '''
        b_size = in_vector.shape[0]
        w_invector = self.Wup(in_vector)    # N*300 --> N*4000
        w_invector = w_invector/tau
        logits = w_invector.reshape(b_size, self.L, self.V)
        p_theta = torch.nn.Softmax(-1)(logits).reshape(b_size, -1)   # reshaped prob-logits
        q_theta = self.Wq(p_theta)  # q only for BYOL
        return logits, p_theta, q_theta
          
    def task_forward(self, batched_data, sem_tau=1.):
        # downstream task forward
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        logits, p_theta, _ = self.SEM(h_graph, sem_tau)    
        output = self.task_head(p_theta)
        return logits, output

    def distill_forward(self, batched_data, sem_tau=1.):
        # for distill, both use logits
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch) 
        logits, p_theta, _ = self.SEM(h_graph, sem_tau)
        return logits, p_theta   
        
    def backbone_forward(self, batched_data):
        # output original N*300 vector
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        return h_graph

    def refgame_forward(self,batched_data, tau=1.):
    # Not ready yet
        h_node = self.gnn_node(batched_data)
        z_out = self.pool(h_node, batched_data.batch) 
        _, _, q_out = self.SEM(z_out, tau)
        return z_out, q_out
    
    def bgrl_forward(self,batched_data):
    # Not ready yet
        h_node = self.gnn_node(batched_data)
        hq_node = self.Wq(h_node)
        return h_node, hq_node
    
    def byol_forward(self, batched_data, tau=1.):
    # Not ready yet
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch) 
        _, p_theta, q_theta = self.SEM(h_graph, tau)
        return p_theta, q_theta

class GNN_SEM_UPDOWN(GNN):
    '''
        For design 2, but use z_post for the task
        the message matrix has shape L*V
        Task head is emb_dim->Ntask
        For distillation, output long L*V vector
        For SSL, need more Wq and Wg
    '''
    def __init__(self, L=200, V=20, tau=1., **kwargs):
        super(GNN_SEM_UPDOWN, self).__init__(**kwargs)
        self.L = L
        self.V = V
        self.Wup = nn.Linear(self.emb_dim, self.L*self.V)
        self.Wdown = nn.Linear(self.L*self.V, self.emb_dim)
        self.Wq = nn.Linear(self.emb_dim, self.emb_dim)
        #self.task_head = nn.Linear(self.emb_dim, self.num_tasks)
        self.task_head = nn.Sequential(
                            nn.Linear(self.emb_dim, self.emb_dim),
                            nn.ReLU(),
                            nn.Linear(self.emb_dim, self.num_tasks),
                            )
        
    def SEM(self, in_vector, tau=1.):
        '''
            Piecewise softmax on a long 1*(L*V) vector
            Use tau to control the softmax temperature
            e.g., embd_size=300, L=30, V=10, as we have 30 words, each with 10 possible choices
        '''
        b_size = in_vector.shape[0]
        w_invector = self.Wup(in_vector)    # N*300 --> N*4000
        w_invector = w_invector/tau
        logits = w_invector.reshape(b_size, self.L, self.V)
        prob = torch.nn.Softmax(-1)(logits).reshape(b_size, -1)   # reshaped prob-logits
        p_theta = self.Wdown(prob)
        q_theta = self.Wq(p_theta)  # q only for BYOL
        return logits, p_theta, q_theta
          
    def task_forward(self, batched_data, sem_tau=1.):
        # downstream task forward
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        logits, p_theta, _ = self.SEM(h_graph, sem_tau)    
        output = self.task_head(p_theta)
        return logits, output

    def distill_forward(self, batched_data, sem_tau=1.):
        # for distill, both use logits
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch) 
        logits, p_theta, _ = self.SEM(h_graph, sem_tau)
        return logits, p_theta   

class GNN_SEM_LSTM(GNN):
    '''
        For design 2, but use z_post for the task
        the message matrix has shape L*V
        Task head is emb_dim->Ntask
        For distillation, output long L*V vector
        For SSL, need more Wq and Wg
    '''
    def __init__(self, L=200, V=20, tau=1., **kwargs):
        super(GNN_SEM_LSTM, self).__init__(**kwargs)
        self.L = L
        self.V = V
        self.tau = tau
        self.ENC = MsgGenLSTM(L=self.L, V=self.V, tau=self.tau,hidden_size=self.emb_dim)
        self.DEC = MsgDecoderLSTM(L=self.L, V=self.V, hidden_size=self.emb_dim)
        self.Wq = nn.Linear(self.emb_dim, self.emb_dim)
        #self.task_head = nn.Linear(self.emb_dim, self.num_tasks)
        self.task_head = nn.Sequential(
                            nn.Linear(self.emb_dim, self.emb_dim),
                            nn.ReLU(),
                            nn.Linear(self.emb_dim, self.num_tasks),
                            )
        
    def SEM(self, in_vector,  tau=1.,  mode='gumbel'):
        '''
            Piecewise softmax on a long 1*(L*V) vector
            Use tau to control the softmax temperature
            e.g., embd_size=300, L=30, V=10, as we have 30 words, each with 10 possible choices
        '''
        b_size = in_vector.shape[0]
        z_pre = in_vector
        msg, logits = self.ENC.forward(z_pre, z_pre, tau=tau, mode=mode)
        p_theta = self.DEC.forward(msg)
        p_theta = p_theta.reshape(b_size,-1)        # [NB, embd]
        q_theta = self.Wq(p_theta)
        return logits, p_theta, q_theta
          
    def task_forward(self, batched_data, sem_tau=1., mode='gumbel'):
        # downstream task forward
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        logits, p_theta, _ = self.SEM(h_graph, sem_tau, mode=mode)   
        output = self.task_head(p_theta)
        return logits, output

    def distill_forward(self, batched_data, sem_tau=1., mode='gumbel'):
        # for distill, both use logits
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch) 
        logits, p_theta, _ = self.SEM(h_graph, sem_tau, mode=mode)
        return logits, p_theta  

class GNN_SEM_GUMBEL(GNN):
    '''
        For design 2, but use z_post for the task
        the message matrix has shape L*V
        Task head is emb_dim->Ntask
        For distillation, output long L*V vector
        For SSL, need more Wq and Wg
    '''
    def __init__(self, L=200, V=20, tau=1., **kwargs):
        super(GNN_SEM_GUMBEL, self).__init__(**kwargs)
        self.L = L
        self.V = V
        self.Wup = nn.Linear(self.emb_dim, self.L*self.V)
        self.word_sel = torch.nn.Linear(self.V,3)
        self.Wdown = nn.Linear(3*self.L, self.emb_dim)       
        self.Wq = nn.Linear(self.emb_dim, self.emb_dim)
        #self.task_head = nn.Linear(self.emb_dim, self.num_tasks)
        self.task_head = nn.Sequential(
                            nn.Linear(self.emb_dim, self.emb_dim),
                            nn.ReLU(),
                            nn.Linear(self.emb_dim, self.num_tasks),
                            )
        
    def SEM(self, in_vector, tau=1.):
        '''
            Piecewise softmax on a long 1*(L*V) vector
            Use tau to control the softmax temperature
            e.g., embd_size=300, L=30, V=10, as we have 30 words, each with 10 possible choices
        '''
        b_size = in_vector.shape[0]
        w_invector = self.Wup(in_vector)    # N*300 --> N*4000
        #w_invector = w_invector/tau
        logits = w_invector.reshape(b_size, self.L, self.V)
        msg_oht = torch.nn.functional.gumbel_softmax(logits,tau=tau, hard=True,dim=-1)
        msg = self.word_sel(msg_oht)  # Shape is N*L
        p_theta = self.Wdown(msg.reshape(b_size,-1))
        q_theta = self.Wq(p_theta)  # q only for BYOL
        return logits, p_theta, q_theta
          
    def task_forward(self, batched_data, sem_tau=1.):
        # downstream task forward
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        logits, p_theta, _ = self.SEM(h_graph, sem_tau)    
        output = self.task_head(p_theta)
        return logits, output

    def distill_forward(self, batched_data, sem_tau=1.):
        # for distill, both use logits
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch) 
        logits, p_theta, _ = self.SEM(h_graph, sem_tau)
        return logits, p_theta 

class SEMPool(nn.Module):
    def __init__(self, emb_dim, L, V, tau):
        super(SEMPool, self).__init__()
        self.linear = nn.Linear(emb_dim, L*V, bias=False)
        self.tau = tau
        self.L = L
        self.V = V

    def forward(self, h, batch_id):
        outs = []
        o = self.linear(h) #[1243, L*V]
        o = gnn.global_add_pool(o, batch_id)
        logits = o.view(-1, self.L, self.V)  #[64, L,V]
        logits = logits/self.tau
        p = F.softmax(logits, -1)
        p = p.view(-1, self.L*self.V)
        return logits, p

class GNN_SEM_POOL(GNN):
    '''
        For design 1, the message matrix has shape L*V
        The pool used here are different
        Task head is L*V->L, then L->Ntask
        For distillation, output long L*V vector
        For SSL, need more Wq and Wg
    '''
    def __init__(self, L=200, V=20, tau=1., **kwargs):
        super(GNN_SEM_POOL, self).__init__(**kwargs)
        self.L = L
        self.V = V
        self.tau=tau
        self.Wq = nn.Linear(self.L*self.V, self.L*self.V)
        self.task_head = nn.Sequential(
                            nn.Linear(self.L*self.V, self.L, bias=False),
                            nn.ReLU(),
                            nn.Linear(self.L, self.num_tasks)
                            )       
        self.sem_pool = SEMPool(self.emb_dim, self.L, self.V, self.tau)

    def task_forward(self, batched_data, sem_tau=1.):
        # downstream task forward
        h_node = self.gnn_node(batched_data)
        logits, p_theta = self.sem_pool(h_node, batched_data.batch)  
        output = self.task_head(p_theta)
        return logits, output

    def distill_forward(self, batched_data, sem_tau=1.):
        # for distill, both use logits
        h_node = self.gnn_node(batched_data)
        logits, p_theta = self.sem_pool(h_node, batched_data.batch) 
        return logits, p_theta


if __name__ == '__main__':
    GNN(num_tasks = 10)