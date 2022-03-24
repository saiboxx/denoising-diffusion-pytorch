#!/bin/sh
#
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=32GB
#SBATCH --account=core-rad
#SBATCH -o logs/slurm/slurm%A_%a.log
#SBATCH -e logs/slurm/slurm%A_%a.err
#SBATCH -J sbatch_single

singularity exec --nv --no-home -B /projects/cds/tobweber/core-demo/:/projects/cds/tobweber/core-demo /projects/containers/2022-03-09_pytorch_workshop.simg "$@"
