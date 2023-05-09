from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
from ogb.lsc import PygPCQM4Mv2Dataset
from ogb.utils import smiles2graph
import os
import numpy as np

PATH = '/home/joshua52/projects/def-dsuth/joshua52/P4_Graph/dataset/'
if not os.path.exists(PATH):
    PATH = 'E:\\P4_Graph\\dataset'

# possible datasets: (size,batch.num_nodes,y.shape)
# molpcba(10949-763-128), molhiv(1029-852-1), molbace (38-1103-1), 
# molbbbp(51-781-1), molclintox(37-858-2), molmuv(2328-764-17), molsider(36-1502-27), moltox21(196-431-12), 
# moltoxcast(215-455-617), molesol(29-417-1), molfreesolv(17-257-1), mollipo(105-783-1)

def build_dataset(args, force_name=None):
    if force_name is not None:
        dataset_name = force_name
    else:
        dataset_name = args.dataset_name
    
    if dataset_name == 'pcqm':
        dataset = PygPCQM4Mv2Dataset(root = PATH, smiles2graph = smiles2graph)
        test_name = 'valid'
    else:
        dataset = PygGraphPropPredDataset(name = dataset_name)
        test_name = 'test'

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]
    
    if args.dataset_forcetask != 0:
        dataset.data.y = dataset.data.y[:,:args.dataset_forcetask]
    split_idx = dataset.get_idx_split()
    ### automatic evaluator. takes dataset name as input
    train_part = dataset[split_idx["train"]]
    if args.dataset_ratio!=1.:
        ratio_idx = int(args.dataset_ratio*len(split_idx["train"]))
        train_part = dataset[split_idx["train"]][:ratio_idx]
    if args.dataset_hardsplit=='hard':
        file_name = args.dataset_name+'_hard_0p8.npy'
        sel_index = np.load(os.path.join(PATH, file_name))
        train_part = dataset[split_idx["train"][sel_index]]
        
    train_loader = DataLoader(train_part, 
                            batch_size=args.batch_size, shuffle=True, drop_last=True,
                            num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], 
                            batch_size=args.batch_size, shuffle=True, drop_last=True,
                            num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx[test_name]], 
                            batch_size=args.batch_size, shuffle=True, drop_last=True,
                            num_workers = args.num_workers)
    if dataset_name == 'pcqm':
        args.num_tasks = 1
        args.task_type = "regression"
        args.eval_metric = "mae"
    else:
        if  args.dataset_forcetask != 0:
            args.num_tasks = 1
            args.task_type = 'binary classification'
            args.eval_metric = 'rocauc'
        else:
            args.num_tasks = dataset.num_tasks
            args.task_type = dataset.task_type
            args.eval_metric = dataset.eval_metric
    loaders = {'train':train_loader, 'valid':valid_loader, 'test':test_loader}
    return loaders