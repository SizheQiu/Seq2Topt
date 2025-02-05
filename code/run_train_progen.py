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
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import warnings
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from tokenizers import Tokenizer

'''
Run the training process. Progen2
'''
def set_random_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def split_data( data, ratio=0.1):
    idx = np.arange(len( data[0]))
    np.random.shuffle(idx)
    num_split = int(len(data[0]) * ratio)
    idx_1, idx_0 = idx[:num_split], idx[num_split:]
    data_0 = [ data[di][idx_0] for di in range(len(data))]
    data_1 = [ data[di][idx_1] for di in range(len(data))]
    return data_0, data_1

def train_eval(model, train_pack, test_pack , dev_pack, device, lr, batch_size, lr_decay, decay_interval, num_epochs ):
    #Load progen2
    progen_model = AutoModelForCausalLM.from_pretrained("/usr/data/Seq2Topt-main/progen2-small", trust_remote_code=True)
    progen_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("/usr/data/Seq2Topt-main/progen2-small", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  

    criterion = F.mse_loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= decay_interval, gamma=lr_decay)
    idx = np.arange(len(train_pack[0]))
    
    min_size = 4
    div_min = max(1, batch_size // min_size)
        
    train_result = {'rmse_train':[],'r2_train':[],'mae_train':[],'rmse_test':[],'r2_test':[],'mae_test':[],\
                   'rmse_dev':[],'r2_dev':[],'mae_dev':[]}
    
    for epoch in range(num_epochs):
        np.random.shuffle(idx)
        model.train()
        predictions, targets = [],[]
        for i in range(math.ceil( len(train_pack[0]) / min_size )):
            batch_data = [train_pack[di][idx[ i* min_size: (i + 1) * min_size]] for di in range(len(train_pack))]
            ids, seqs, y = batch_data
            seqs = list(map(str, seqs))  
            batch_tokens = tokenizer(
                seqs, padding="max_length",
                truncation=True, max_length=1024, return_tensors="pt").to(device)

            input_ids = batch_tokens["input_ids"]
            attention_mask = batch_tokens["attention_mask"]

            with torch.no_grad():
                outputs = progen_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                emb = outputs.hidden_states[-1]  # last layer hidden state
            emb = emb.transpose(1,2) # (batch, features, seqlen)
            emb = emb.to(device)
            
            target_values = torch.FloatTensor( np.array( [ np.array([v]) for v in y ] ) )
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
        
#         if r2_test > 0.52:
#             win_size = model.window
#             print('Best model found at epoch=' + str(epoch) + '!')
#             best_model_pth = '../data/model_topt_window='+str(win_size)+'r2=' +str(r2_test)+'.pth'
#             torch.save( model.state_dict(), best_model_pth)
        
        if epoch%5 == 0:
            print('epoch: '+str(epoch)+'/'+ str(num_epochs) +';  rmse test: ' + str(rmse_test) + '; r2 test: ' + str(r2_test) )
        
        scheduler.step()
        
    return train_result

def test(model, test_pack,  batch_size, device ):
    progen_model = AutoModelForCausalLM.from_pretrained("/usr/data/Seq2Topt-main/progen2-small", trust_remote_code=True)

    progen_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("/usr/data/Seq2Topt-main/progen2-small", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  
    
    model.eval()
    predictions, target_values = [],[]
    for i in range(math.ceil( len(test_pack[0]) / batch_size )):
        batch_data = [test_pack[di][i * batch_size: (i + 1) * batch_size] for di in range(len(test_pack))]
        ids, seqs, y = batch_data
        seqs = list(map(str, seqs))  
        batch_tokens = tokenizer(
            seqs, padding="max_length", truncation=True,
            max_length=1024, return_tensors="pt" ).to(device)

        input_ids = batch_tokens["input_ids"]
        attention_mask = batch_tokens["attention_mask"]
        with torch.no_grad():
            outputs = progen_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            emb = outputs.hidden_states[-1]
        emb = emb.transpose(1,2) # (batch, features, seqlen)
        emb = emb.to(device)
        
        with torch.no_grad():
            preds = model( emb )
        predictions += preds.cpu().detach().numpy().reshape(-1).tolist()
        target_values += list(y)     
    
    predictions = np.array(predictions)
    target_values = np.array(target_values)
    rmse = get_rmse( target_values, predictions)
    r2 = get_r2( target_values, predictions)
    mae = get_mae( target_values, predictions)
    return rmse, r2, mae



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--task',choices=['topt','tm','pHopt'], default='topt')
    parser.add_argument('--train_path', default='/usr/data/PredOT-main/PredOT-main-1/PredOT-main/data/Topt/train_os.csv')
    parser.add_argument('--test_path', default='/usr/data/PredOT-main/PredOT-main-1/PredOT-main/data/Topt/test.csv')

    parser.add_argument('--lr', default = 0.0005, type=float )
    parser.add_argument('--batch', default = 32 , type=int )
    parser.add_argument('--lr_decay', default = 0.5, type=float )
    parser.add_argument('--decay_interval', default = 10, type=int )
    parser.add_argument('--num_epoch', default = 30, type=int ) #30
    parser.add_argument('--param_dict_pkl', default = '/usr/data/PredOT-main/PredOT-main-1/PredOT-main/data/hyparams/default.pkl')

    args = parser.parse_args()
    
    set_random_seeds(0)
    
    train_path, test_path, lr, batch_size, lr_decay, decay_interval = \
            str(args.train_path), str(args.test_path), float(args.lr), int(args.batch), \
            float(args.lr_decay), int(args.decay_interval)
    
    task = str(args.task)
    print('The task is '+ task+'!')
   
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    rparams = {'topt':(0,120),'tm':(0,100),'pHopt':(0,14)}
    train_pack = [np.array(train_data.index), np.array(train_data.sequence), \
              np.array( rescale_targets(list(train_data[task]), rparams[task][1], rparams[task][0])) ]
    test_pack = [np.array(test_data.index), np.array(test_data.sequence), \
             np.array( rescale_targets(list(test_data[task]), rparams[task][1], rparams[task][0])) ]

    train_pack, dev_pack = split_data( train_pack, 0.1)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('We use CUDA!')
    else:
        device = torch.device('cpu')
        print('We use CPU!')
    
    
    num_epochs = int( args.num_epoch )
    warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
    
    emb_dim= 1024  # progen2
    n_head = 4; n_RD = 4;
    # for win_size in [3,5,7]:
    win_size = 7
    M = MultiAttModel( emb_dim, win_size, n_head, n_RD)
    M.to(device)

    train_result = train_eval( M , train_pack, test_pack , dev_pack, device, lr, batch_size, lr_decay,\
                    decay_interval,  num_epochs )
    train_result['Epoch'] = list(np.arange(1,num_epochs+1))
    result_pd = pd.DataFrame( train_result )
    output_path = os.path.join(  '/usr/data/Seq2Topt-main/data/output',task +'_progen2_window'+str(win_size)+'.csv' )

    result_pd.to_csv(output_path,index=None)

    print('Done.')

















