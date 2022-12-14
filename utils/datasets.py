from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader


# possible datasets: (size,batch.num_nodes,y.shape)
# molpcba(10949-763-128), molhiv(1029-852-1), molbace (38-1103-1), 
# molbbbp(51-781-1), molclintox(37-858-2), molmuv(2328-764-17), molsider(36-1502-27), moltox21(196-431-12), 
# moltoxcast(215-455-617), molesol(29-417-1), molfreesolv(17-257-1), mollipo(105-783-1)

def build_dataset(args, force_name=None):
    if force_name is not None:
        dataset = PygGraphPropPredDataset(name = force_name)
    else:
        dataset = PygGraphPropPredDataset(name = args.dataset_name)
    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]
    split_idx = dataset.get_idx_split()
    ### automatic evaluator. takes dataset name as input
    train_loader = DataLoader(dataset[split_idx["train"]], 
                               batch_size=args.batch_size, shuffle=True, 
                               num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], 
                               batch_size=args.batch_size, shuffle=True, 
                               num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], 
                               batch_size=args.batch_size, shuffle=True, 
                               num_workers = args.num_workers)
    args.num_tasks = dataset.num_tasks
    args.task_type = dataset.task_type
    args.eval_metric = dataset.eval_metric
    loaders = {'train':train_loader, 'valid':valid_loader, 'test':test_loader}
    return loaders