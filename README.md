# Seq2Topt
A deep learning model of enzyme optimal temperature

## Datasets
sequence_ogt_topt.csv obtained from https://github.com/jafetgado/tomer.<br>
pH opt data obtained from EpHod: https://zenodo.org/records/8011249.<br>
Tm data obtained from https://github.com/liimy1/DeepTM/tree/master/Data.<br>
## How to use:
1. Prepare the input file: a CSV file containing a column "sequence" for protein sequences.<br>
2. Enter `/code` and run prediction: <br>
```
python seq2topt.py --input [input.csv] --output [output file name]
```
The same for seq2tm.py and seq2pHopt.py in `/code`.
## Dependency:
1.Pytorch: https://pytorch.org/<br>
2.ESM: https://github.com/facebookresearch/esm<br>
3.Scikit-learn: https://scikit-learn.org/<br>
4.Seaborn statistical data visualization:https://seaborn.pydata.org/index.html<br>
## Citation
