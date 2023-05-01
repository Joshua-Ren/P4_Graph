import wandb
import argparse
import numpy as np
import os
import copy
import torch.optim as optim
import toml
from enegine_toy_3dshapes import train_epoch, train_distill, evaluate
from utils.general import update_args, wandb_init, get_init_net_toy, rnd_seed, AverageMeter, early_stop_meets
from utils.nil_related import *
from utils.toy_example import generate_3dshape_fullloader_vae
from models.vae import BetaVAE_H

def get_args_parser():
    # Training settings
    # ======= Usually default settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--config_file', type=str, default=None,
                        help='the name of the toml configuration file')
    parser.add_argument('--seed', default=10086, type=int)
    parser.add_argument('--WD_ID',default='joshuaren', type=str,
                        help='W&D ID, joshuaren or joshua_shawn')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    # ======== Dataset and task related
    parser.add_argument('--dataset_name', default='3dshape', type=str,
                        help='3dshapes or dsprite')    
    parser.add_argument('--sup_ratio', default=0.1, type=float,
                        help='ratio of the training factors')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size of train and test set')
    parser.add_argument('--num_class', default=1, type=int,
                        help='How many reg-tasks, 1~6')
    parser.add_argument('--learning_rate', default=1e-3, type=float,
                        help='LR for beta-VAE')
    parser.add_argument('--beta', default=3, type=float,
                        help='beta in beta-vae')
    parser.add_argument('--recon_loss', default='gaussian', type=str,
                        help='gaussian for 3dshapes, bernoulli for dsprites')

    # ===== Wandb and saving results ====
    parser.add_argument('--run_name',default='beta_vae_pretrain',type=str)
    parser.add_argument('--proj_name',default='P4_toy', type=str)    
    return parser

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum')
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    else:
        recon_loss = None

    return recon_loss

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

def main(args):
    ALPHA_TABLE = [1., 0.3, 0.1, 0.03, 0.01, 0.001]
    # ========== Generate seed ==========
    results = {'tloss':[],'vloss':[], 'dis_loss':[]}
    if args.seed==0:
        args.seed = np.random.randint(1,10086)
    rnd_seed(args.seed)

    # ========== Prepare save folder and wandb ==========
    run_name = wandb_init(proj_name=args.proj_name, run_name=args.run_name, config_args=args)
    args.save_path = os.path.join('results','toy_betavae',run_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    seed_net = BetaVAE_H(z_dim=10,nc=3)
    img_seed = None
    for alpha in ALPHA_TABLE:
        args.sup_ratio = alpha
        # ========== Prepare the loader and optimizer
        full_loader = generate_3dshape_fullloader_vae(args)
        net = copy.deepcopy(seed_net)
        net.to(args.device)
        optimizer = optim.Adam(net.parameters(),lr=args.learning_rate)
        for i,(x,_,reg,idx) in enumerate(full_loader):
            x = x.float().to(args.device)
            if img_seed is None:
                img_seed = x[:8]
            x_recon, mu, logvar = net(x)
            recon_loss = reconstruction_loss(x, x_recon, args.recon_loss)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            beta_vae_loss = recon_loss + args.beta*total_kld
            optimizer.zero_grad()
            beta_vae_loss.backward()
            optimizer.step()
            # ------ Report to wandb
            wandb.log({'idx_epoch':i})
            wandb.log({'recon_loss':recon_loss.item()})
            if i%300==0:
                x_recon_seed, _, _ = net(img_seed)
                images = wandb.Image(x_recon_seed.cpu().detach(), caption='epoch_'+str(i)+'_in_alpha_'+str(alpha))
                wandb.log({"recon": images})
                
        save_name = 'bvae_alpha_'+str(alpha)+'.pth'
        save_path = os.path.join(args.save_path, save_name)
        torch.save(net.state_dict(),save_path)
        del full_loader
if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    args.device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    main(args)