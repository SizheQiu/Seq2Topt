#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=hyopt_tm
#SBATCH --time=10:00:00 
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx8000:1

module load PyTorch/1.7.1-fosscuda-2020b

samples=("1" "2" "3" "4" "5" "6" "7" "8" "9")
for index in ${samples[@]}
do
echo $index
start_time=$(date +%s)

eval "python run_train.py --task tm --train_path ../data/Tm/train_os_tm.csv --test_path ../data/Tm/test_tm.csv \
    --param_dict_pkl ../data/hyparams/params_${index}.pkl"
    
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo $elapsed
done