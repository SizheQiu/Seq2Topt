# Seq2Topt
A deep learning model of enzyme optimal temperature.

## Datasets
sequence_ogt_topt.csv obtained from https://github.com/jafetgado/tomer.<br>
pH opt data obtained from EpHod: https://zenodo.org/records/8011249.<br>
Tm data obtained from https://github.com/liimy1/DeepTM/tree/master/Data.<br>
## Accuracy:
1. Seq2Topt: RMSE = 13.3℃ and R2=0.48<br>
2. Seq2Tm: RMSE=7.57℃ and R2=0.64<br>
3. Seq2pHopt: RMSE=0.92 and R2=0.37<br>
## How to use:
1. Prepare the input file: a CSV file containing a column "sequence" for protein sequences.<br>
2. Enter `/code` directory and run prediction: <br>
```
python seq2topt.py --input [input.csv] --output [output file name]
```
The same for seq2tm.py and seq2pHopt.py in `/code`.<br>
3. Hyperparameters and model pth files: <br>
	- Seq2Topt: `/data/hyparams/best_topt_param.pkl` and `/data/model_pth/model_topt_r2test=0.5002.pth`<br>
	- Seq2Tm: `/data/hyparams/default.pkl` and `../data/model_pth/model_Tm_r2=0.682152.pth` <br>
	- Seq2pHopt: `/data/hyparams/default.pkl` and `../data/model_pth/model_pHopt_rmse=0.064849.pth`<br>
4. Feel free to use `/code/model.py` to develop other predictive models for proteins. <br>
## Workflow:
1. Model evaluation: `/code/Model_evaluation.ipynb`
2. Selection of thermophilic enzymes: `/code/CaseStudy_thermophile.ipynb`
3. Analysis of residue attention weights: `/code/AnalysisResidueAttention.ipynb`
4. Prediction of optimal temperature shifts: `/code/CaseStudy_mutations.ipynb`
## Dependency:
1.Pytorch: https://pytorch.org/<br>
2.ESM: https://github.com/facebookresearch/esm<br>
3.Scikit-learn: https://scikit-learn.org/<br>
4.Seaborn statistical data visualization:https://seaborn.pydata.org/index.html<br>
## Citation
Qiu, S., Hu, B., Zhao, J., Xu, W., Yang, A. (2024). Seq2Topt: A Sequence-Based Deep Learning Predictor of Enzyme Optimal Temperature. doi:10.1101/2024.08.12.607600 
