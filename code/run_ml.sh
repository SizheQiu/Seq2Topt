#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=DL_GPU
#SBATCH --clusters=htc
#SBATCH --time=10:00:00 
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx8000:1



module load PyTorch/1.7.1-fosscuda-2020b


start_time=$(date +%s)
python run_train.py --task topt --train_path ../data/Topt/train_os.csv --test_path ../data/Topt/test.csv
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo $elapsed
echo "Topt Done!"


start_time=$(date +%s)
python run_train.py --task tm --train_path ../data/Tm/Tm_Train.csv --test_path ../data/Tm/Tm_Test.csv
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo $elapsed
echo "Tm Done!"


start_time=$(date +%s)
python run_train.py --task pHopt --train_path ../data/pHopt/train_pH.csv --test_path ../data/pHopt/test_pH.csv
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo $elapsed
echo "pHopt Done!"



