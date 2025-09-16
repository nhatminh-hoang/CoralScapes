#!/bin/bash

#SBATCH --job-name=SSP_CoralScapes
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodelist=hpc24
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-gpu=100G

#SBATCH --output=train_outs/gpu/out/%x.%j.out
#SBATCH --error=train_outs/gpu/errors/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=21013299@st.phenikaa-uni.edu.vn

conda init bash
source ~/.bashrc
conda activate gpu_11.8

srun python train.py