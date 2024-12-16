#!/bin/bash
#
#SBATCH --partition=disc
#SBATCH --ntasks=1
#SBATCH --mem=62G
#SBATCH --output=jobname_%J_stdouttanhtrial1updatedtrial12.txt
#SBATCH --error=jobname_%J_stderrtanhtrial1updatedtrial12.txt
#SBATCH --time=47:00:00
#SBATCH --job-name=scalarfile
#SBATCH --mail-user=sujata.sahu-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/ssahu
#SBATCH --gres=gpu:1
#
module load Python/3.11.5-GCCcore-13.2.0
module load cuDNN/8.9.7.29-CUDA-12.3.0
source $HOME/naccs/bin/activate
export CUDA_DIR=$CUDA_HOME
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_HOME

# Run the Python script
python $HOME/tanhtrial1updatedtrial12.py


