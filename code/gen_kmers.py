import os
import numpy as np
import pandas as pd
import argparse
import sys
import pickle
from sklearn.metrics import r2_score, mean_absolute_error
from functions import *


def split_sequence(sequence, ngram, word_dict):
    sequence = '>' + sequence + '<'
    words = [ word_dict[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)]
    return np.array(words)

def get_kmers(data_path, output_path, task, ngram, dict_path, has_target ):
    data = pd.read_csv(data_path)
    word_dict = load_pickle( dict_path )
    proteins, topts = [],[]
    T_max, T_min = 120.0, 0.0
    for i in data.index:
        seq = list(data['sequence'])[i]
        proteins.append(split_sequence( seq, ngram,  word_dict ))
        if has_target:
            topts.append( np.array([ list(data['topt'])[i] ]) )
        
    dump_pickle( proteins, os.path.join(output_path, task+'_proteins.pkl') )    
    if has_target:
        norm_targets = rescale_targets( topts, T_max, T_min)
        dump_pickle( norm_targets, os.path.join(output_path, task+'_normtopts.pkl') )
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate 3-mers for protein sequences. \
                        --data_path: the input path of data (csv);\
                        --output_path: the output path of 3-mers;\
                        --task: train or test;\
                        --dict_path: the path of 3-mer dictionary;\
                        --has_target: whether there is target column in input data, True or False.')

    parser.add_argument('--data_path',  required = True)
    parser.add_argument('--output_path',  required = True)
    parser.add_argument('--task', type=str, choices=['train','test'], required = True)
    parser.add_argument('--dict_path',  default = '../data/word_dict.pkl')
    parser.add_argument('--has_target',  type=str, choices=['False','True'], default = 'True')
    args = parser.parse_args()
    
    ngram = 3 #use 3-mer
    
    data_path, output_path, task, dict_path = str(args.data_path), str(args.output_path),\
                str(args.task), str(args.dict_path)
    if str(args.has_target)=='True':
        has_target=True
    else:
        has_target=False
        
    get_kmers(data_path, output_path, task, ngram, dict_path, has_target )
    
    print('Completed.')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
