import os
import numpy as np
import sys
import pickle
from sklearn.metrics import r2_score, mean_absolute_error
import torch
import math
from math import sqrt
import requests
from urllib import request
import html
from collections import Counter

def get_seq(ID):
    '''
    Query protein sequence from uniprot database.
    '''
    url = "https://www.uniprot.org/uniprot/%s.fasta" % ID
    try :
        data = requests.get(url)
        if data.status_code != 200:
            seq = 'NaN'
        else:
            seq =  "".join(data.text.split("\n")[1:])
    except :
        seq = 'NaN'
    return seq

def split_table( table, ratio ):
    idx=list(table.index)
    np.random.shuffle(idx)
    num_split = int( len(idx) * ratio)
    idx_test, idx_train = idx[:num_split], idx[num_split:]
    train_table = (table.iloc[idx_train]).reset_index().drop(['index'],axis=1)
    test_table = (table.iloc[idx_test]).reset_index().drop(['index'],axis=1)
    return train_table,test_table


'''
Three evaluation methods.
'''
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




def get_aac(seq):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    output = {}
    for aa in AA:
        output['AAC_'+aa] = seq.count(aa)/len(seq)
    return output

def get_dpc(seq):
    '''
    dipeptide composition (DPC)
    '''
    output = {}
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    DPs = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    for dp in DPs:
        output['DPC_'+ dp] = seq.count(dp)/(len(seq)-1)
    return output

def get_aacdpc(seq):
    if 'X' in seq:
        most_aa = Counter(seq).most_common()[0][0]
        if most_aa == 'X':
            most_aa = Counter(seq).most_common()[1][0]
        seq = seq.replace('X',most_aa)
    results = get_aac(seq)
    results.update( get_dpc(seq) )
    return results
        
    
    
    
    
    
    
    
    
    
    
    
    
        
