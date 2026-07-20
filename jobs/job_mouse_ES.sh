#!/bin/bash
#SBATCH --job-name=scCGNet_mouse_ES     # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=16          # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G                   # memory per cpu-core (4G is default)
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=20:25:50             # total run time limit (HH:MM:SS)
#SBATCH --account=rrg-hup           # rrg-hup - // def-hup-ab
#SBATCH --output=logs/mouse_ES.out      # standard output file
#SBATCH --error=logs/mouse_ES.txt       # standard error file

module load apptainer

cd /home/moctard/scratch/scGMCM-VGAE/
mkdir -p logs

apptainer exec --nv \
  --bind $PWD:/workspace \
  /home/moctard/scratch/scrna2.sif \
  python3 /workspace/main.py --dataset_name=mouse_ES
