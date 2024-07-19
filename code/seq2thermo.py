import math
import pickle
import numpy as np
import pandas as pd
import argparse
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from functions import *
from model import MultiAttModel
import os
import warnings
import random
import esm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inputs: --input: the path of input dataset(csv); \
                                    --output: output path of prediction result.')
    
    parser.add_argument('--input', required = True)
    parser.add_argument('--output', required = True)
    args = parser.parse_args()
    
    tm_pth = ''; topt_pth = '';
    params_tm = load_pickle( ''); params_topt = load_pickle( '');
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU!')
    else:
        device = torch.device('cpu')
        print('CPU!')
        
    emb_dim= 1280  # esm2_t33_650M_UR50D
    model_topt = MultiAttModel( emb_dim, device, window, n_head, dropout, n_RD)
    model_topt.to(device);
    model_topt.load_state_dict(torch.load( topt_pth, map_location=device  ))
    
    model_tm = MultiAttModel( emb_dim, device, window, n_head, dropout, n_RD)
    model_tm.to(device);
    model_tm.load_state_dict(torch.load( tm_pth, map_location=device  ))
    
    input_data = pd.read_csv(str(args.input))
    #Load esm2
    esm2_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D() # 33 layers
    esm2_model = esm2_model.to(device)
    esm2_batch_converter = alphabet.get_batch_converter()
    
    
    result_pd = pd.DataFrame(zip(IDs,seqs,pred_topts, pred_tms), 
                             columns=['id','sequence','pred_topt','pred_tm'])
    
    result_pd.to_csv( str( args.output ) +'.csv' ,index=None)
    print('Task '+ str(args.input)+' completed!')
    
    
    
    
    