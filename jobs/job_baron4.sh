#!/bin/bash
#SBATCH --job-name=scCGNet_Baron4    # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=28:25:50             # total run time limit (HH:MM:SS)
#SBATCH --account=rrg-hup           # rrg-hup - // def-hup-ab
#SBATCH --output=logs/Baron4.out      # standard output file
#SBATCH --error=logs/Baron4.txt       # standard error file

module load apptainer

cd /home/moctard/scratch/scGMCM-VGAE/
mkdir -p logs

apptainer exec --nv \
  --bind $PWD:/workspace \
  /home/moctard/scratch/scrna2.sif \
  python3 /workspace/main.py --dataset_name=baron4
