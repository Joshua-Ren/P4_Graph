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
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from models.conv import GNN_node, GNN_node_Virtualnode

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
                                            nn.Linear(2*self.emb_dim, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, self.num_tasks)
                                            )
                                            
        else:
            #self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)
            self.graph_pred_linear = nn.Sequential(
                                            nn.Linear(self.emb_dim, 128),
                                            nn.ReLU(),
                                            nn.Linear(128, 128),
                                            nn.ReLU(),                                      
                                            nn.Linear(128, self.num_tasks)
                                            )
    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        return self.graph_pred_linear(h_graph)
        
    def dis_foward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)   
        return h_graph



class GNN_SEM_BYOL(GNN):
    '''
        This is for another design, let's try BYOL first
    '''
    def __init__(self, L=200, V=20, **kwargs):
        super(GNN_SEM_BYOL, self).__init__(**kwargs)
        self.L = L
        self.V = V
        self.Wup = nn.Linear(self.emb_dim, self.L*self.V)
        self.Wg = nn.Linear(self.L*self.V, self.emb_dim)
        self.Wq = nn.Linear(self.emb_dim, self.emb_dim)
        self.task_head = nn.Sequential(
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
        w_invector = w_invector/tau
        logits = w_invector.reshape(b_size, self.L, self.V)
        y_theta = torch.nn.Softmax(-1)(logits).reshape(b_size, -1)   # N*4000
        z_theta = self.Wg(y_theta)
        q_theta = self.Wq(z_theta)
        return logits, y_theta, z_theta, q_theta
          
    def task_forward(self, batched_data, tau=1.):
        # downstream task forward
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        _, y_theta, _, _ = self.SEM(h_graph, tau)    
        output = self.task_head(y_theta)
        return output
    
    def ssl_forward(self, batched_data, tau=1.):
        # for BYOL, online use q_theta, target use z_theta
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch) 
        _, _, z_theta, q_theta = self.SEM(h_graph, tau)
        return z_theta, q_theta

    def distill_forward(self, batched_data, tau=1.):
        # for distill, both use logits
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch) 
        logits, _, _, _ = self.SEM(h_graph, tau)
        return logits        
        
    def backbone_forward(self, batched_data):
        # output original N*300 vector
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        return h_graph

class GNN_SEM(GNN):
    '''
        This is for another design, let's try BYOL first
    '''
    def __init__(self, mlp_hidden=128, tau=1.,gum_tau=1., L=10, V=30, **kwargs):
        super(GNN_SEM, self).__init__(**kwargs)
        self.mlp_hidden = mlp_hidden
        self.tau = tau
        self.gum_tau = gum_tau
        self.L = L
        self.V = V
        self.pre_SEM = torch.nn.Linear(self.emb_dim, self.L*self.V)
        self.word_sel = torch.nn.Linear(self.V,3)
        self.after_SEM = torch.nn.Linear(3*self.L, self.emb_dim)
        self.SEM_after = torch.nn.Linear(self.L*self.V, self.L*self.V)
        
    def SEM(self, in_vector):
        '''
            Piecewise softmax on a long 1*(L*V) vector
            Use tau to control the softmax temperature
            e.g., embd_size=300, L=30, V=10, as we have 30 words, each with 10 possible choices
        '''
        b_size, emb_dim = in_vector.shape[0], in_vector.shape[1]
        #w_invector = self.SEM_lin(in_vector)
        w_invector = in_vector
        w_invector = w_invector/self.tau
        logits = w_invector.reshape(b_size, self.L, self.V)
        msg_oht = torch.nn.functional.gumbel_softmax(logits,tau=self.gum_tau, hard=True,dim=-1)
        msg = self.word_sel(msg_oht)  # Shape is N*L
        dis_hidden = self.after_SEM(msg.reshape(b_size,-1))   # Use for task
        #tmp_softmax = torch.nn.Softmax(2)(w_invector)
        #out_vector = tmp_softmax.reshape(b_size, -1)
        return dis_hidden, logits
          
    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        dis_hidden, logits = self.SEM(h_graph)    
        output = self.graph_pred_linear(dis_hidden)
        return output
    
    def backbone_forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)
        return h_graph
 
    def msg_foward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)   
        dis_hidden, logits = self.SEM(h_graph)  # N*(L*V)
        return dis_hidden, logits
        
    def dis_foward(self, batched_data):
        # Similar to msg_forward, only for the old code
        h_node = self.gnn_node(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)   
        dis_h_graph = self.SEM(h_graph)
        return dis_h_graph

if __name__ == '__main__':
    GNN(num_tasks = 10)