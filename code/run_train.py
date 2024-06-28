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
from model import PredOT
import os
import warnings
import random
import esm


def train_eval(model, train_pack, test_pack , dev_pack, device, lr, batch_size, lr_decay, decay_interval, num_epochs ):
    #Load esm2
    esm2_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D() # 6 layers
    esm2_model = esm2_model.to(device)
    esm2_batch_converter = alphabet.get_batch_converter()
    
    criterion = F.mse_loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= decay_interval, gamma=lr_decay)
    idx = np.arange(len(train_pack[0]))
    
    min_size = 4
    if batch_size > min_size:
        div_min = int(batch_size / min_size)
        
    train_result = {'rmse_train':[],'r2_train':[],'mae_train':[],\
                    'rmse_test':[],'r2_test':[],'mae_test':[],\
                   'rmse_dev':[],'r2_dev':[],'mae_dev':[]}
    
    for epoch in range(num_epochs):
        np.random.shuffle(idx)
        model.train()
        predictions, targets = [],[]
        for i in range(math.ceil( len(train_pack[0]) / min_size )):
            batch_data = [train_pack[di][idx[ i* min_size: (i + 1) * min_size]] for di in range(len(train_pack))]
            ids, seqs, topts = batch_data
            #Generate emb
            input_data = [(ids[i], seqs[i]) for i in range(len(ids))]
            batch_labels, batch_strs, batch_tokens = esm2_batch_converter(input_data)
            batch_tokens = batch_tokens.to(device=device, non_blocking=True)
            with torch.no_grad():
                emb = esm2_model(batch_tokens, repr_layers=[6], return_contacts=False)
            emb = emb["representations"][6]
            emb = emb.transpose(1,2) # (batch, features, seqlen)
            emb = emb.to(device)
         
            target_values = torch.FloatTensor( np.array( [ np.array([topt]) for topt in topts ] ) )
            target_values = target_values.to(device)
            
            pred = model( emb )
            loss = criterion(pred.float(), target_values.float())
            predictions += pred.cpu().detach().numpy().reshape(-1).tolist()
            targets += target_values.cpu().numpy().reshape(-1).tolist()
            loss.backward()
            if i % div_min == 0 and i != 0:    
                optimizer.step()
                optimizer.zero_grad()
                
        predictions = np.array(predictions)
        targets = np.array(targets)
        train_result['rmse_train'].append( get_rmse( targets, predictions) )
        train_result['r2_train'].append( get_r2( targets, predictions) )
        train_result['mae_train'].append( get_mae( targets, predictions) )
        
        rmse_test, r2_test, mae_test = test(model, test_pack, batch_size, device )
        rmse_dev, r2_dev, mae_dev = test(model, dev_pack, batch_size, device )
        train_result['rmse_test'].append(rmse_test); train_result['r2_test'].append(r2_test); train_result['mae_test'].append(mae_test);
        train_result['rmse_dev'].append(rmse_dev); train_result['r2_dev'].append(r2_dev); train_result['mae_dev'].append(mae_dev);
        
        if epoch%5 == 0:
            print('epoch: '+str(epoch)+'/'+ str(num_epochs) +';  rmse test: ' + str(rmse_test) + '; r2 test: ' + str(r2_test) )
        
        scheduler.step()
        
    return train_result
            
def test(model, test_pack,  batch_size, device ):
    #Load esm2
    esm2_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D() # 6 layers
    esm2_model = esm2_model.to(device)
    esm2_batch_converter = alphabet.get_batch_converter()
    
    model.eval()
    predictions, target_values = [],[]
    ids, seqs, targets = test_pack
    for i in range(math.ceil( len(train_pack[0]) / batch_size )):
        batch_data = [test_pack[di][idx[ i* batch_size: (i + 1) * batch_size]] for di in range(len(test_pack))]
        ids, seqs, targets = batch_data
        #Generate emb
        input_data = [(ids[i], seqs[i]) for i in range(len(ids))]
        batch_labels, batch_strs, batch_tokens = esm2_batch_converter(input_data)
        batch_tokens = batch_tokens.to(device=device, non_blocking=True)
        with torch.no_grad():
            emb = esm2_model(batch_tokens, repr_layers=[6], return_contacts=False)
        emb = emb["representations"][6]
        emb = emb.transpose(1,2) # (batch, features, seqlen)
        emb = emb.to(device)
        with torch.no_grad():
            preds = model( emb )
        predictions += preds.cpu().detach().numpy().reshape(-1).tolist()
        target_values += list(targets)  
            
    
    predictions = np.array(predictions)
    target_values = np.array(target_values)
    rmse = get_rmse( target_values, predictions)
    r2 = get_r2( target_values, predictions)
    mae = get_mae( target_values, predictions)
    return rmse, r2, mae
    

    
def split_data( data, ratio=0.1):
    idx = np.arange(len( data[0]))
    np.random.shuffle(idx)
    num_split = int(len(data[0]) * ratio)
    idx_1, idx_0 = idx[:num_split], idx[num_split:]
    data_0 = [ data[di][idx_0] for di in range(len(data))]
    data_1 = [ data[di][idx_1] for di in range(len(data))]
    return data_0, data_1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train_path', required = True)
    parser.add_argument('--test_path', required = True)
    parser.add_argument('--lr', default = 0.001, type=float )
    parser.add_argument('--batch', default = 32 , type=int )
    parser.add_argument('--lr_decay', default = 0.5, type=float )
    parser.add_argument('--decay_interval', default = 10, type=int )
    parser.add_argument('--num_epoch', default = 30, type=int )
    parser.add_argument('--param_dict_pkl', default = '../data/performances/default.pkl')
    args = parser.parse_args()
    
    train_path, test_path, lr, batch_size, lr_decay, decay_interval, param_dict_pkl = \
            str(args.train_path), str(args.test_path), float(args.lr), int(args.batch), \
            float(args.lr_decay), int(args.decay_interval) , str( args.param_dict_pkl )
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    train_data, dev_data = split_table( train_data, 0.1 )
    
    T_max, T_min = 120, 0
    train_pack = [np.array(train_data.uniprot_id), np.array(train_data.sequence), \
                  np.array( rescale_targets(list(train_data.topt), T_max, T_min)) ];
    test_pack = [np.array(test_data.uniprot_id), np.array(test_data.sequence), \
                 np.array( rescale_targets(list(test_data.topt), T_max, T_min)) ];
    dev_pack = [np.array(dev_data.uniprot_id), np.array(dev_data.sequence), \
                np.array( rescale_targets(list(dev_data.topt), T_max, T_min)) ];

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('We use CUDA!')
    else:
        device = torch.device('cpu')
        print('We use CPU!')
    
    
    num_epochs = int( args.num_epoch )
    param_dict = load_pickle(param_dict_pkl)
    
    window, dropout, layer_cnn, layer_output = \
            param_dict['window'],param_dict['dropout'],param_dict['layer_cnn'],param_dict['layer_out']
    warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
    
    dim=320  # esm2_t6_8M_UR50D
    M = PredOT( dim, device, window, dropout, layer_cnn, layer_output)
    M.to(device);
    
    train_result = train_eval( M , train_pack, test_pack , dev_pack, device, lr, batch_size, lr_decay,\
                   decay_interval,  num_epochs )
    train_result['Epoch'] = list(np.arange(1,num_epochs+1))
    result_pd = pd.DataFrame( train_result )
    output_path = os.path.join(  '../data/performances/',os.path.basename(param_dict_pkl).split('.')[0] + \
                               '_lr=' + str(lr) + '_batch='+str(batch_size)+ \
                               '_lr_decay=' + str(lr_decay) + '_decay_interval=' + str(decay_interval) +'.csv' )
    
    result_pd.to_csv(output_path,index=None)
    
    print('Done.')
    
    
    
        
        
        
        
        
        
        
        
