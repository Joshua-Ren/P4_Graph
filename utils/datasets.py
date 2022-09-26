from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader

def build_dataset(args):
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
                               batch_size=args.batch_size, shuffle=False, 
                               num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], 
                               batch_size=args.batch_size, shuffle=False, 
                               num_workers = args.num_workers)
    args.num_tasks = dataset.num_tasks
    args.task_type = dataset.task_type
    args.eval_metric = dataset.eval_metric
    loaders = {'train':train_loader, 'valid':valid_loader, 'test':test_loader}
    return loaders