#!/bin/sh
#
#SBATCH --gpus=2
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32GB
#SBATCH --account=core-rad
#SBATCH -o logs/slurm/slurm%A_%a.log
#SBATCH -e logs/slurm/slurm%A_%a.err
#SBATCH --array=1-25
#SBATCH -J sbatch_multi

# get file names
file=$(ls configs/ | head -n $SLURM_ARRAY_TASK_ID | tail -n 1)
echo "parsing config: "$file
singularity exec --nv --no-home -B /projects/cds/tobweber/core-demo/:/projects/cds/tobweber/core-demo /projects/containers/2022-03-09_pytorch_workshop.simg python examples/gpu/03_cifar_ddp_fp16.py -c configs/$file
