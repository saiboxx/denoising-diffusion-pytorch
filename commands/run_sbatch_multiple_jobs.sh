#!/bin/sh
#
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=50GB
#SBATCH --account=core-rad
#SBATCH -o logs/slurm/slurm%A_%a.log
#SBATCH -e logs/slurm/slurm%A_%a.err
#SBATCH --array=1-15%1
#SBATCH -J sbatch_multi

singularity exec \
  --nv \
  --no-home \
  -B /projects/core-rad/tobweber/ddpm,/projects/core-rad/data/ILSVRC2012_img_train \
  /projects/containers/ngc-pytorch:21.09-py3 \
  commands/init_multi_2.sh $SLURM_ARRAY_TASK_ID
