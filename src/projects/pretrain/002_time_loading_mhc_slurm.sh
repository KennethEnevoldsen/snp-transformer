#!/bin/bash
#SBATCH --mem 64G
#SBATCH -t 2:0:0
#SBATCH -c 4
#SBATCH -A NLPPred
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=slurm-notifications-aaaahkuvjxiplokhffpn6qphzu@chcaa.slack.com
#SBATCH --output ./docs/slurm-output/%x-%u-%j.out

echo "starting"
python src/projects/pretrain/time_loading_mhc.py