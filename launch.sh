#!/bin/bash

module load cuda/11.2
module load python/3.8
source ~/env_1.12/bin/activate

wandb agent lavoiems/sem_gnn/aee1bve1
