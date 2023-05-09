#!/bin/bash
#SBATCH --account=def-dsuth
#SBATCH --gres=gpu:v100l:1              # Number of GPU(s) per node
#SBATCH --cpus-per-task=8         # CPU cores/threads
#SBATCH --mem=64000M               # memory per node
#SBATCH --time=1-00:00            # time (DD-HH:MM)
#SBATCH --output=./logs/stage1.txt 


# 1. Load the required modules
module load python/3.8

# 2. Load your environment
source /home/joshua52/projects/def-dsuth/joshua52/env_graph/bin/activate

# 3. Copy your dataset on the compute node
#cp /network/datasets/<dataset> $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR

cd /home/joshua52/projects/def-dsuth/joshua52/P4_Graph

#srun python /home/joshua52/projects/def-dsuth/joshua52/P4_Graph/main_nil.py \
#--config_file hiv_gcn_semnil_hard



srun python /home/joshua52/projects/def-dsuth/joshua52/P4_Graph/main_nil.py \
--config_file hiv_gcn_baseline_hard