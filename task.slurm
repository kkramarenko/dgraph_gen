#!/bin/bash

#SBATCH --nodes=1 --ntasks-per-node=8
#SBATCH --job-name=test

cd $SLURM_SUBMIT_DIR
python test.py
