from torch import nn
import torch.nn.functional as F
from torch_geometric import nn as gnn


class SEMPool(nn.Module):
    def __init__(self, emb_dim, L, V, tau):
        super(SEMPool, self).__init__()
        self.linear = nn.Linear(emb_dim, L*V, bias=False)
        self.tau = tau
        self.L = L
        self.V = V

    def forward(self, h, batch_id):
        outs = []
        o = self.linear(h)
        o = gnn.global_add_pool(o, batch_id)
        o = o.view(-1, self.L, self.V)
        o = F.softmax(o/self.tau, -1)
        o = o.view(-1, self.L*self.V)

        return o

