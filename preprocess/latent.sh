#!/bin/bash -l

#SBATCH --partition=gpu-a100
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --ntasks=16
#SBATCH --qos=long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiuntian001@ntu.edu.sg
#SBATCH --hint=nomultithread
#SBATCH --gpus=1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=latent-1024-000-099

module load miniconda
module load cuda

source activate /home/user/jiuntian/.conda/envs/breakascene

ls -1 /home/user/jiuntian/data/sa1b/sa_000{000..099}.tar | sort | xargs -P 2 -I {} python encode_latents.py {}