#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=DL_hyopt
#SBATCH --clusters=htc
#SBATCH --time=10:00:00 
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx8000:1

module load PyTorch/1.7.1-fosscuda-2020b

list1=("3" "5" "7")
list2=("2" "3" "4" "5")

train_path=../data/Topt/train_os.csv
test_path=../data/Topt/test.csv

start_time=$(date +%s)
for n_RD in "${list2[@]}"; do
	eval "python run_train.py --task topt --train_path ${train_path} --test_path ${test_path} \
        --param_dict_pkl ../data/hyparams/w3nRD${n_RD}.pkl"
done
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo $elapsed
echo "Done!"


#start_time=$(date +%s)
#for n_RD in "${list2[@]}"; do
#	eval "python run_train.py --task topt --train_path ${train_path} --test_path ${test_path} \
#        --param_dict_pkl ../data/hyparams/w5nRD${n_RD}.pkl"
#done
#end_time=$(date +%s)
#elapsed=$(( end_time - start_time ))
#echo $elapsed
#echo "Done!"


#start_time=$(date +%s)
#for n_RD in "${list2[@]}"; do
#	eval "python run_train.py --task topt --train_path ${train_path} --test_path ${test_path} \
#        --param_dict_pkl ../data/hyparams/w7nRD${n_RD}.pkl"
#done
#end_time=$(date +%s)
#elapsed=$(( end_time - start_time ))
#echo $elapsed
#echo "Done!"





