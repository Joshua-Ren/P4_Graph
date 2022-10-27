#!/bin/bash
#SBATCH -A NLP-CDT-SL2-GPU
#SBATCH -p ampere
#SBATCH --job-name=pcba_nil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=10G
#SBATCH --output=./logs/stage1.txt 
#SBATCH --gres=gpu:1

# 1. Load the required modules
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.8 cuda/11.0 cudnn/8.0_cuda-11.1

# 2. Load your environment
source /home/sg955/glm-env/bin/activate

# 3. Copy your dataset on the compute node
#cp /network/datasets/<dataset> $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR

cd /home/sg955/GitWS/P4_Graph/

srun python main_nil.py --WD_ID joshua_shawn \
--proj_name P4_phase_observe --dataset_name ogbg-molpcba --batch_size 107 \
--backbone_type gcn --bottle_type upsample --L 123 --V 30 \
--drop_ratio 0 --scheduler False --dis_loss ce_argmax \
--dis_lr 0.0007565782660579647 --dis_sem_tau 1 --dis_smp_tau 2 --epochs_dis 5 --teach_last_best best --dis_loss ce_argmax \
--epochs_ft 1000 --es_epochs 4 --ft_lr 0.000106491 --ft_tau 0.834511539 \
--run_name nil_gcn_pcba_dp0_cemax_earlystp_besthyper