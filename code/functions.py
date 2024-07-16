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

# def get_entry_ot( ec, brenda ):
#     r = brenda.reactions.get_by_id(ec)
#     all_data = r.temperature['optimum']
#     result = []
#     for i in range(len(all_data)):
#         OT = all_data[i]['value']
#         if '#' not in all_data[i]['meta']:
#             continue
#         p_refs = []
#         if ';' in all_data[i]['meta']:
#             meta_list = all_data[i]['meta'].split(';')
#             for meta in meta_list:
#                 p_refs += list( meta.split('#')[1].split(',') )
                
#         else:
#             meta = all_data[i]['meta']
#             p_refs += list( meta.split('#')[1].split(',') )
            
#         for ref in p_refs:
#             if (ref in r.proteins) and (r.proteins[ref]['proteinID'] != ''):
#                 p_id = r.proteins[ref]['proteinID']
#                 p_id = p_id.replace('UniProt','').replace('SwissProt','').\
#                 replace('swissprot','').replace('Uniprot','').replace('TrEMBL','').replace('GenBank','').strip()
#                 if ' and ' in p_id:
#                     p_ids = p_id.split(' and ')
#                     for p_id in p_ids:
#                         result.append( {'uniprot_id': p_id.strip(),'topt':float(OT)} )        
#                 elif ' ' in p_id:
#                     p_id = max( p_id.split(' '), key=len )
#                     result.append( {'uniprot_id': p_id.strip(),'topt':float(OT)} )
#                 else:
#                     result.append( {'uniprot_id': p_id.strip(),'topt':float(OT)} )    
#     return result


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
    
    
# def load_pkl2ndarr(data_path):
#     '''
#     Load pickle data (seq k-mers and targets) as nd array.
#     '''
#     ndarr = np.array( load_pickle( data_path ), dtype=object )
#     return ndarr    
    
# def data2tensor( batch_data, has_target, device ):
#     words = batch_data[0]
#     N = max([a.shape[0] for a in words])
#     temp_arr = np.zeros((len(words), N)) # add padding for protein k-mers
#     for i, a in enumerate( words ):
#         n = a.shape[0]
#         temp_arr[i, :n] = a + 1
#     words = torch.LongTensor(temp_arr).to(device)
    
#     if has_target:
#         targets = batch_data[1]
#         temp_arr = np.zeros((len(targets), 1))
#         for i,a in enumerate( targets ):
#             temp_arr[i, :] = a
#         targets = torch.FloatTensor(temp_arr).to(device)
#     else:
#         return words
    
#     return words, targets
        
    
    
    
    
    
    
    
    
    
    
    
    
        
