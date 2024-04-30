#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=DL_GPU
#SBATCH --clusters=htc
#SBATCH --time=10:00:00 
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx8000:1



module load PyTorch/1.7.1-fosscuda-2020b

train_path=../data/train_os.csv
test_path=../data/test.csv


start_time=$(date +%s)

python run_train.py --train_path ${train_path} --test_path ${test_path} --lr 0.0005

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo $elapsed
echo "Done!"
