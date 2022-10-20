#!/bin/bash
#SBATCH --partition=long                           # Ask for unkillable job
#SBATCH --cpus-per-task=2                                # Ask for 2 CPUs
#SBATCH --gres=gpu:1                                     # Ask for 1 GPU
#SBATCH --mem=10G                                        # Ask for 10 GB of RAM
#SBATCH --job-name=nil
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

srun python /home/mila/y/yi.ren/P4_Graph/main_resethead.py \
--drop_ratio 0.5 --L 200 --V 20 \
--proj_name P4_phase_observe --dataset_name ogbg-moltox21 \
--backbone_type gcn --bottle_type gumbel \
--epochs_ft 100 --generations 20 \
--run_name reset_tox_gcn_gumbel_2lhead_dp05