# Seq2Topt
A deep learning model of enzyme optimal temperature.

## Datasets
sequence_ogt_topt.csv obtained from https://github.com/jafetgado/tomer.<br>
pH opt data obtained from EpHod: https://zenodo.org/records/8011249.<br>
Tm data obtained from https://github.com/liimy1/DeepTM/tree/master/Data.<br>
## Accuracy:
1. Seq2Topt: RMSE = 12.26℃ and R2=0.57 <br>
2. Seq2Tm: RMSE=7.57℃ and R2=0.64 <br>
3. Seq2pHopt: RMSE=0.88 and R2=0.42 <br>
## How to use:
1. Download model weights from [Release](https://github.com/SizheQiu/Seq2Topt/releases/tag/v1.0.0) of this repo.
2. Model hyperparameters: dim=320, window=3, n_head=4, n_RD=4.
3. Follow the [tutorial notebook](https://github.com/SizheQiu/Seq2Topt/blob/main/code/Tutorial.ipynb) to try Seq2Topt model.
4. Feel free to use `/code/model.py` to develop other predictive models for proteins. <br>
## Workflow:
1. Model evaluation: `/code/Model_evaluation.ipynb`
2. Selection of thermophilic enzymes: `/code/CaseStudy_thermophile.ipynb`
3. Analysis of residue attention weights: `/code/AnalysisResidueAttention.ipynb`
4. Prediction of optimal temperature shifts: `/code/CaseStudy_mutations.ipynb`
## Dependency:
1.Pytorch: https://pytorch.org/<br>
2.ESM: https://github.com/facebookresearch/esm<br>
3.ProGen2: https://github.com/salesforce/progen
4.Scikit-learn: https://scikit-learn.org/<br>
5.Seaborn statistical data visualization:https://seaborn.pydata.org/index.html<br>
## Citation
Qiu, S., Hu, B., Zhao, J., Xu, W., Yang, A. (2024). Seq2Topt: A Sequence-Based Deep Learning Predictor of Enzyme Optimal Temperature. doi:10.1101/2024.08.12.607600 
