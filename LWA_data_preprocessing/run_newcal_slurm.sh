#!/bin/bash
#SBATCH --job-name=newcal
#SBATCH --partition=nointerrupt
# BATCH --nodes=1  #activate to request exclusive use of a node
# BATCH --ntasks-per-node=10  #for exlusive node request. number of cores
#SBATCH --mem=300G
#SBATCH --time=336:00:00
#SBATCH --output=/home/rbyrne/slurm_newcal.out
#SBATCH --export=ALL

cd ~
source ~/.bashrc
conda activate py310
date
python /opt/devel/rbyrne/rlb_LWA/LWA_data_preprocessing/run_newcal.py $1

