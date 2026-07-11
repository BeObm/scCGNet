#!/bin/bash
#SBATCH --job-name=scCGNet_Shekar    # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=16           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=16G            # memory per cpu-core (4G is default)
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=49:10:50             # total run time limit (HH:MM:SS)
#SBATCH --account=def-hup-ab          # rrg-hup - // def-hup-ab 
#SBATCH --output=output_Shekar.txt
#SBATCH --error=error_Shekar.txt
# Standard error file


module load apptainer

cd /home/moctard/scratch/scGMCM-VGAE/

apptainer exec --nv \
  --bind $PWD:/workspace \
  /home/moctard/scratch/scrna2.sif \
  python3 /workspace/main.py --dataset_name=Shekar