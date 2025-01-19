#!/bin/bash -l

#SBATCH --partition=cpu-epyc-genoa
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --nodes=1
#SBATCH --mem=8G
#SBATCH --ntasks=2
#SBATCH --qos=long
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiuntian001@ntu.edu.sg
#SBATCH --hint=nomultithread
#SBATCH --time=1-00:00:00
#SBATCH --job-name=extract-ori-size

module load miniconda
# module load cuda

source activate /home/user/jiuntian/.conda/envs/breakascene

ls -1 /home/user/jiuntian/data/sa1b/sa_000{500..999}.tar | sort | xargs -P 16 -I {} python extract_sdxl_ori_size.py {}