#!/bin/bash
#SBATCH --account=rrg-dsuth
#SBATCH --gres=gpu:v100l:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --mem=80000M               # memory per node
#SBATCH --time=1-10:00            # time (DD-HH:MM)
#SBATCH --output=./logs/stage1.txt 
#SBATCH --job-name=toy


# 1. Load the required modules
module load python/3.8

# 2. Load your environment
source /home/joshua52/projects/def-dsuth/joshua52/env_graph/bin/activate

# 3. Copy your dataset on the compute node
#cp /network/datasets/<dataset> $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR

cd /home/joshua52/projects/def-dsuth/joshua52/P4_Graph

srun python /home/joshua52/projects/def-dsuth/joshua52/P4_Graph/main_generate_betavae.py \
--sup_ratio 0.5 --run_name beta_vae_pretrain_alpha0p5