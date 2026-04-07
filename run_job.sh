#!/bin/bash
#SBATCH --job-name=scgmcm       # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=8           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G            # memory per cpu-core (4G is default)
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --time=02:01:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-type=END               # Send email at job completion
#SBATCH --account=rrg-hup
#SBATCH --output=old_model.txt
#SBATCH --error=error.txt             # Standard error file


module load python

pip install --no-index -r requirements.txt

python old_model.py