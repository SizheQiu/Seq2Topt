import os
import numpy as np
import sys
import pickle
from sklearn.metrics import r2_score, mean_absolute_error
import torch
import math
from math import sqrt

def get_rmse(x,y):
    rmse = sqrt(((x - y)**2).mean(axis=0))
    return round(rmse, 6)

def get_r2(x,y):
    return round( r2_score(x,y), 6)

def get_mae(x,y):
    return  round( mean_absolute_error(x,y), 6 )

def run( cmd ):
    os.system( cmd )
    return None

def load_pickle(filename):
    temp = None
    with open(filename,'rb') as f:
        temp = pickle.load(f)
    return temp
        
def dump_pickle(file, filename):
    with open(filename, 'wb') as f:
        pickle.dump( file , f)

        
def rescale_targets(target_values, x_max, x_min):
    return [ (x-x_min)/(x_max-x_min) for x in target_values]


# def gen_worddict(seq_list, ngram ):
#     word_dict = {}
#     k = 0
#     for seq in seq_list:
#         seq = '>' + seq + '<'
#         for i in range(len(seq)-ngram+1):
#             kmer = seq[i:i+ngram]
#             if kmer not in word_dict:
#                 if len(word_dict.keys()) == 0:
#                     word_dict[kmer] = 0
#                 else:
#                     word_dict[kmer] = max(list(word_dict.values())) + 1
#         k += 1
#         if k%1000 == 0:
#             print( str( round( k/len(seq_list),4) ) + ' completed.' )
                    
#     return word_dict 
    
    
def load_pkl2ndarr(data_path):
    '''
    Load pickle data (seq k-mers and targets) as nd array.
    '''
    ndarr = np.array( load_pickle( data_path ), dtype=object )
    return ndarr    
    
def data2tensor( batch_data, has_target, device ):
    words = batch_data[0]
    N = max([a.shape[0] for a in words])
    temp_arr = np.zeros((len(words), N)) # add padding for protein k-mers
    for i, a in enumerate( words ):
        n = a.shape[0]
        temp_arr[i, :n] = a + 1
    words = torch.LongTensor(temp_arr).to(device)
    
    if has_target:
        targets = batch_data[1]
        temp_arr = np.zeros((len(targets), 1))
        for i,a in enumerate( targets ):
            temp_arr[i, :] = a
        targets = torch.FloatTensor(temp_arr).to(device)
    else:
        return words
    
    return words, targets
        
    
    
    
    
    
    
    
    
    
    
    
    
        
