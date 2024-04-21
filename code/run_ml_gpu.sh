#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=DL_GPU
#SBATCH --clusters=htc
#SBATCH --time=10:00:00 
#SBATCH --partition=short
#SBATCH --gres=gpu:v100:1



module load PyTorch/1.7.1-fosscuda-2020b

input=../data/input/


start_time=$(date +%s)

python run_train.py --input_path ${input} --lr 0.0005

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo $elapsed
echo "Done!"
