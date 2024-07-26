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

'''
Predict enzyme optimal pH.
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict enzyme optimal pH from protein sequences. \
                                    Inputs: --input: the path of input dataset(csv); \
                                    --output: output path of prediction result.')
    
    parser.add_argument('--input', required = True)
    parser.add_argument('--output', required = True)
    args = parser.parse_args()
    
    pHopt_pth = '../data/model_pth/model_pHopt_rmse=0.064849.pth';
    params = load_pickle( '../data/hyparams/default.pkl' );
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU!')
    else:
        device = torch.device('cpu')
        print('CPU!')
    emb_dim= 320
    window, dropout, n_head, n_RD = \
            params['window'],params['dropout'],params['n_head'],params['n_RD']
    warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
    
    model = MultiAttModel( emb_dim, device, window, n_head, dropout, n_RD)
    model.to(device);
    model.load_state_dict(torch.load( pHopt_pth, map_location=device  ))
    model.eval()
    
    input_data = pd.read_csv(str(args.input))
    batch_size=4
    #Load esm2
    esm2_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D() # 6 layers
    esm2_model = esm2_model.to(device)
    esm2_batch_converter = alphabet.get_batch_converter()
    predictions = []
    for i in range( math.ceil( len(input_data.index) / batch_size ) ):
        ids = list(input_data.index)[i * batch_size: (i + 1) * batch_size]
        seqs = list(input_data['sequence'])[i * batch_size: (i + 1) * batch_size]
        #embeddings
        inputs = [(ids[i], seqs[i]) for i in range(len(ids))]
        batch_labels, batch_strs, batch_tokens = esm2_batch_converter(inputs)
        batch_tokens = batch_tokens.to(device=device, non_blocking=True)
        with torch.no_grad():
            emb = esm2_model(batch_tokens, repr_layers=[6], return_contacts=False)
        emb = emb["representations"][6]
        emb = emb.transpose(1,2)
        emb = emb.to(device)
    
        with torch.no_grad():
            preds = model( emb )
        predictions += preds.cpu().detach().numpy().reshape(-1).tolist()
    
    pred_pHs = [float(v*14) for v in predictions ]
    result_pd = pd.DataFrame(zip(list(input_data.index), list(input_data['sequence']), pred_pHs ),\
                             columns=['id','sequence','pred_pHopt'])
    result_pd.to_csv( str( args.output ) +'.csv' ,index=None)
    print('Task '+ str(args.input)+' completed!')
    
    
    
    
    
    
    
    
    