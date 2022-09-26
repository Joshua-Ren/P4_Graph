import torch
import copy
import numpy as np
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.transforms import Compose
# ========== Message and its evaluations ===============
def cal_msg_distance_model(args, student, teacher, batch):
  with torch.no_grad():
    student.eval()
    teacher.eval()
    stud_logits = student.distill_forward(batch)
    teach_logits = teacher.distill_forward(batch)
    dist = (stud_logits.argmax(-1)==teach_logits.argmax(-1)).sum()/(stud_logits.shape[0]*stud_logits.shape[1])
    return dist

def cal_msg_distance_logits(logits1, logits2):
  with torch.no_grad():
    dist = (logits1.argmax(-1)==logits2.argmax(-1)).sum()/(logits1.shape[0]*logits1.shape[1])
    return dist

# ============= Graph data augmentation =============
    # ------- Copy from BGRL: https://github.com/nerdslab/bgrl/blob/main/bgrl/transforms.py
class DropFeatures:
    r"""Drops node features with probability p."""
    def __init__(self, p=None, precomputed_weights=True):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, data):
        drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
        data.x[:, drop_mask] = 0
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class DropEdges:
    r"""Drops edges with probability p."""
    def __init__(self, p, force_undirected=False):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p

        self.p = p
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        edge_attr = data.edge_attr if 'edge_attr' in data else None

        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.p, force_undirected=self.force_undirected)

        data.edge_index = edge_index
        if edge_attr is not None:
            data.edge_attr = edge_attr
        return data

    def __repr__(self):
        return '{}(p={}, force_undirected={})'.format(self.__class__.__name__, self.p, self.force_undirected)


def get_graph_drop_transform(drop_edge_p, drop_feat_p):
    transforms = list()

    # make copy of graph
    transforms.append(copy.deepcopy)

    # drop edges
    if drop_edge_p > 0.:
        transforms.append(DropEdges(drop_edge_p))

    # drop features
    if drop_feat_p > 0.:
        transforms.append(DropFeatures(drop_feat_p))
    return Compose(transforms)


# ============= Model parameters update =============
class EMA():
  def __init__(self, eta):
    super().__init__()
    self.eta = eta

  def update_average(self, old, new):
    if old is None:
      return new
    return old * self.eta + (1 - self.eta) * new
    
def EMA_update(online_model, target_model, eta=0.99):
  ema = EMA(eta)
  with torch.no_grad():
    for online_params, target_params in zip(online_model.parameters(),target_model.parameters()):
      old_weight, up_weight = target_params.data, online_params.data
      target_params.data = ema.update_average(old_weight, up_weight)