#!/bin/bash
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:rtx8000:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --job-name=nil-hiv
#SBATCH --time=40:00:00                                   # The job will run for 3 hours
#SBATCH --output=./logs/stage1.txt 

# 1. Load the required modules
module load python/3.8

# 2. Load your environment
source /home/mila/y/yi.ren/env_graph/bin/activate

# 3. Copy your dataset on the compute node
#cp /network/datasets/<dataset> $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR

cd /home/mila/y/yi.ren/P4_Graph/

srun python /home/mila/y/yi.ren/P4_Graph/main_nil.py \
--drop_ratio 0 \
--proj_name P4_phase_observe --dataset_name ogbg-molhiv --distill_set ogbg-molhiv \
--backbone_type gcn --bottle_type upsample --drop_ratio 0 --L 200 --V 20 \
--epochs_ft 100 --steps_dis 5000 --generations 10 --ft_lr 0.0002 \
--dis_loss ce_sample --dis_smp_tau 1 --teach_last_best last --dis_optim_type adam --dis_lr 0.001 \
--run_name hiv_distill_hiv_train_last_1en4ftlr