from ogb.lsc import PCQM4Mv2Dataset
from ogb.utils import smiles2graph

dataset = PCQM4Mv2Dataset(root = './', smiles2graph = smiles2graph)