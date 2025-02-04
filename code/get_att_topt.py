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
from model import RDBlock
import os
import warnings
import random
import esm

class ModelAtt(nn.Module):
    def __init__(self, dim, window, n_head, n_RD):
        super(ModelAtt, self).__init__()
        self.n_RD = n_RD
        self.n_head = n_head
        self.cnn_v = nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window)
        self.W_cnns = nn.ModuleList([ nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window) for _ in range(n_head)])
        self.RDs = nn.ModuleList([RDBlock(2*n_head*dim) for _ in range(n_RD)])  
        self.output = nn.Linear(2*n_head*dim, 1)
        
    def forward(self, emb):
        values = self.cnn_v(emb)
        w_atts = []
        for i in range( self.n_head ):
            weights = F.softmax(self.W_cnns[i](emb), dim=-1)
            x_sum = torch.sum(values * weights, dim=-1) # Sum pooling
            x_max,_ = torch.max(values * weights, dim=-1) # Max pooling
            if i == 0:
                cat_xsum = x_sum
                cat_xmax = x_max
            else:
                cat_xsum = torch.cat([cat_xsum, x_sum],dim=1)
                cat_xmax = torch.cat([cat_xmax, x_max],dim=1)
            w_atts.append( torch.mean(weights,dim=1).cpu().detach().numpy() )
            
        w_atts = np.array(w_atts)
        avg_ws = np.mean( w_atts, axis=0)  
        return avg_ws
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict enzyme Topt from protein sequences. \
                                    Inputs: --input: the path of input dataset(csv); \
                                    --output: output path of prediction result.')
    
    parser.add_argument('--input', required = True)
    parser.add_argument('--output', required = True)
    args = parser.parse_args()
    topt_pth = '../../large_model_pth/model_topt_window=3_r2=0.539321.pth';
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU!')
    else:
        device = torch.device('cpu')
        print('CPU!')
    emb_dim= 320; window=3; n_head = 4; n_RD = 4;
    warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
    
    model = ModelAtt( emb_dim, window, n_head, n_RD)
    model.to(device);
    model.load_state_dict(torch.load( topt_pth, map_location=device  ))
    model.eval()
    
    input_data = pd.read_csv(str(args.input))
    batch_size=4
    #Load esm2
    esm2_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D() # 6 layers
    esm2_model = esm2_model.to(device)
    esm2_batch_converter = alphabet.get_batch_converter()
    avg_weights = []
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
            avg_ws = model( emb )
        avg_weights += list(avg_ws)

    
    avg_weights = np.array( avg_weights )
    dump_pickle( avg_weights, str( args.output ) +'.pkl' )
    print('Task '+ str(args.input)+' completed!')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

