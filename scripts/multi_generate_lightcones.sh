#!/bin/bash -l
# submit array job
#$ -t 1-4
#$ -P darkcosmo
#$ -N gen_fisher_lightcones
#$ -m bae
#$ -M ebaker@bu.edu
#$ -pe omp 16
#$ -l mem_per_core=16G
#$ -l h_rt=12:00:00
#$ -j y

conda activate DM21cm

python /projectnb/darkcosmo/dark_photon_project/21cmfish/scripts/make_lightcones_for_fisher.py /projectnb/darkcosmo/dark_photon_project/21cmfish/21cmFAST_config_files/dark_photon$SGE_TASK_ID.config --h_PEAK=0 --num_cores=1
