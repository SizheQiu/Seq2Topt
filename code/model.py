import torch
import torch.nn as nn
import torch.nn.functional as F




class RDBlock(nn.Module):
    '''A dense layer with residual connection'''
    def __init__(self, dim, dropout ):
        super(RDBlock, self).__init__()
        self.dense = nn.Linear(dim, dim)
        self.batchnorm = nn.BatchNorm1d(dim)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x0 = x
        x = self.activation( self.batchnorm( self.dense(x) ) )        
        x = self.dropout(x)
        x = x0 + x
        return x



class PredOT(nn.Module):
    def __init__(self, dim, device, window, dropout, layer_cnn, layer_output):
        super(PredOT, self).__init__()
        self.device = device
        self.dim = dim
        self.layer_output = layer_output
        self.layer_cnn = layer_cnn
        
#         self.esm2_model, self.esm2_batch_converter = self.load_ESM2_model()
        self.Vconvs = nn.ModuleList([ nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window) for _ in range(layer_cnn)])
        self.Wconvs = nn.ModuleList([ nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window) for _ in range(layer_cnn)])    
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim) for _ in range(layer_output)])
        self.W_pred = nn.Linear(2*dim, 1)
        
        
#     def load_ESM2_model(self):
#         model, alphabet = esm.pretrained.esm2_t33_650M_UR50D() # 33 layers
#         model = model.half() # float16
#         model = model.to(self.device)
#         batch_converter = alphabet.get_batch_converter()
#         return model, batch_converter
    
#     def get_ESM2_embeddings(self, ids, seqs):
#         data = [(ids[i], seqs[i]) for i in range(len(ids))]
#         batch_labels, batch_strs, batch_tokens = self.esm2_batch_converter(data)
#         batch_tokens = batch_tokens.to(device=self.device, non_blocking=True)
#         with torch.no_grad():
#             emb = self.esm2_model(batch_tokens, repr_layers=[33], return_contacts=False)
#         emb = emb["representations"][33]
#         emb = emb.transpose(1,2) # (batch, features, seqlen)
#         emb = emb.to(self.device)
#         return emb
    
    def get_values(self,x):
        for i in range(self.layer_cnn):
            x = F.leaky_relu( self.Vconvs[i](x) )
        return x
    
    
    def get_weights(self,x):
        for i in range(self.layer_cnn):
            x = F.leaky_relu( self.Wconvs[i](x) )
        return x
    
    
    def forward(self, emb):
        values = self.get_values(emb) # CNN
        weights = self.get_weights(emb) # CNN
        weights = F.softmax(weights, dim=-1)
        xa = values * weights # Attention weighted features
        xa_mean = torch.mean( xa, dim=-1) # Mean pooling
        xa_max, _ = torch.max( xa, dim=-1) # Max pooling
        y = torch.cat([ xa_mean, xa_max ], dim=1) # Concat features for regression
        
        for j in range(self.layer_output):
            y =  F.leaky_relu( self.W_out[j](y) )
            
        return self.W_pred(y)
    


class MultiAttModel(nn.Module):
    def __init__(self, dim, device, window, n_head, dropout, n_RD):
        super(MultiAttModel, self).__init__()
        self.n_RD = n_RD
        self.cnn_v = nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window)
        self.W_cnns = nn.ModuleList([ nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window) for _ in range(n_head)])
        self.batchnorm = nn.BatchNorm1d(2*n_head*dim)
        self.dropout = nn.Dropout(dropout)
        self.RDs = nn.ModuleList([RDBlock(2*n_head*dim, dropout) for _ in range(n_RD)])  
        self.output = nn.Linear(2*n_head*dim, 1)
        
    def forward(self, emb):
        values = self.cnn_v(emb)
        for i in range( n_head ):
            weights = F.softmax(self.W_cnns[i](emb) ,dim=-1)
            x_sum = torch.sum(values * weights, dim=-1) # Sum pooling
            x_max,_ = torch.max(values * weights, dim=-1) # Max pooling
            if i == 0:
                cat_xsum = x_sum
                cat_xmax = x_max
            else:
                cat_xsum = torch.cat([cat_xsum, x_sum],dim=1)
                cat_xmax = torch.cat([cat_xmax, x_max],dim=1)
                
        cat_f = torch.cat([ cat_xsum, cat_xmax ], dim=1) # Concat features for regression
        cat_f = self.batchnorm(cat_f)
        cat_f = self.dropout(cat_f)
        
        for j in range(self.n_RD):
            cat_f = self.RDs(cat_f)
            
        return self.output(cat_f)
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
