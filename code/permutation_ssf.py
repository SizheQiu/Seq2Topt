import numpy as np
import pandas as pd
import torch
from model import Seq2Opt


def permute_ssf( ssf, ind ):
    #TODO
    new_ssf = ssf
    return new_ssf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--task',choices=['topt','pHopt','tm'], required = True)
    parser.add_argument('--input', required = True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('We use CUDA!')
    else:
        device = torch.device('cpu')
        print('We use CPU!')


    N_test = 5;
    sig_ssfs = list( load_pickle('../data/sig_ssfs.pkl') )
    emb_dim= 1280  # esm2_t33_650M_UR50D
    M = Seq2Opt( emb_dim, len(sig_ssfs), device, window, n_head, dropout, n_RD)
    M.to(device);

    test_data = pd.read_csv( args.input )
    test_ssf =  load_pickle( os.path.join(os.path.dirname( args.input ),'test_ssf.pkl') )
    results = []
    for i in range( len(sig_ssfs) ):
        temp = { 'name':sig_ssfs[i] }
        for j in range(N_test):
            new_ssf = permute_ssf( test_ssf, i )



