#!/bin/bash
#SBATCH --job-name=fftvis_sim
#SBATCH --partition=general
# BATCH --nodes=1  #activate to request exclusive use of a node
# BATCH --ntasks-per-node=10  #for exlusive node request. number of cores
#SBATCH --mem=100G
#SBATCH --time=336:00:00
#SBATCH --output=/home/rbyrne/slurmtest.out
#SBATCH --export=ALL

cd ~
source ~/.bashrc
conda activate py310
date
python /home/rbyrne/rlb_LWA/LWA_data_preprocessing/generate_model_vis_fftvis.py $1 $2 $3 $4 

