#!/bin/bash
#SBATCH --mem 64G
#SBATCH -t 1:0:0
#SBATCH -c 8
#SBATCH --gres=gpu:1 
#SBATCH -p gpu
#SBATCH -A NLPPred
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=slurm-notifications-aaaahkuvjxiplokhffpn6qphzu@chcaa.slack.com
#SBATCH --output ./docs/slurm-output/%x-%u-%j.out

python src/projects/pretrain/pretrain.py