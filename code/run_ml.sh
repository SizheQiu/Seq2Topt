#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=DL_GPU
#SBATCH --time=10:00:00 
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx8000:1

module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0


echo "Hyp-params opt!"
start_time=$(date +%s)

python run_train.py --task topt --train_path ../data/Topt/train_os.csv --test_path ../data/Topt/test.csv
    
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo $elapsed


# start_time=$(date +%s)
# python run_train.py --task tm --train_path ../data/Tm/Tm50_Train.csv --test_path ../data/Tm/Tm50_Test.csv
# end_time=$(date +%s)
# elapsed=$(( end_time - start_time ))
# echo $elapsed
# echo "Tm Done!"

# start_time=$(date +%s)
# python run_train.py --task pHopt --train_path ../data/pHopt/train_pH.csv --test_path ../data/pHopt/test_pH.csv
# end_time=$(date +%s)
# elapsed=$(( end_time - start_time ))
# echo $elapsed
# echo "Tm Done!"






