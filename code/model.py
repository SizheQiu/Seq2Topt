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
    
    


class MultiAttModel(nn.Module):
    def __init__(self, dim, device, window, n_head, dropout, n_RD):
        super(MultiAttModel, self).__init__()
        self.n_RD = n_RD
        self.n_head = n_head
        self.cnn_v = nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window)
        self.W_cnns = nn.ModuleList([ nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window) for _ in range(n_head)])
        self.batchnorm = nn.BatchNorm1d(2*n_head*dim)
        self.dropout = nn.Dropout(dropout)
        self.RDs = nn.ModuleList([RDBlock(2*n_head*dim, dropout) for _ in range(n_RD)])  
        self.output = nn.Linear(2*n_head*dim, 1)
        
    def forward(self, emb):
        values = self.cnn_v(emb)
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
                
        cat_f = torch.cat([ cat_xsum, cat_xmax ], dim=1) # Concat features for regression
        cat_f = self.batchnorm(cat_f)
        cat_f = self.dropout(cat_f)
        for j in range(self.n_RD):
            cat_f = self.RDs[j](cat_f)
            
        return self.output(cat_f)
            
 
                    
        
# class Seq2Opt(nn.Module):
#     '''
#     Seq2Opt: Multi-head attention, ESM2 embedding + statistical sequence features
#     '''
#     def __init__(self, emb_dim, ssf_dim, device, window, n_head, dropout, n_RD):
#         super(Seq2Opt, self).__init__()
#         self.n_RD = n_RD
#         self.n_head = n_head
#         self.ssf_dim = ssf_dim 
#         self.cnn_v = nn.Conv1d( emb_dim, emb_dim, kernel_size=2*window+1, padding=window)
#         self.W_cnns = nn.ModuleList([ nn.Conv1d( emb_dim, emb_dim, kernel_size=2*window+1, padding=window) for _ in range(n_head)])
#         self.batchnorm = nn.BatchNorm1d(2*n_head*emb_dim + self.ssf_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.RDs = nn.ModuleList([RDBlock(2*n_head*emb_dim + self.ssf_dim, dropout) for _ in range(n_RD)])  
#         self.output = nn.Linear(2*n_head*emb_dim + self.ssf_dim, 1)
        
#     def forward(self, emb, ssf):
#         values = self.cnn_v(emb)
#         for i in range( self.n_head ):
#             weights = F.softmax(self.W_cnns[i](emb), dim=-1)
#             x_sum = torch.sum(values * weights, dim=-1) # Sum pooling
#             x_max,_ = torch.max(values * weights, dim=-1) # Max pooling
#             if i == 0:
#                 cat_xsum = x_sum
#                 cat_xmax = x_max
#             else:
#                 cat_xsum = torch.cat([cat_xsum, x_sum],dim=1)
#                 cat_xmax = torch.cat([cat_xmax, x_max],dim=1)
                
#         cat_f = torch.cat([ cat_xsum, cat_xmax, ssf], dim=1) # Concat features for regression
#         cat_f = self.batchnorm(cat_f)
#         cat_f = self.dropout(cat_f)
#         for j in range(self.n_RD):
#             cat_f = self.RDs[j](cat_f)
            
#         return self.output(cat_f)
            




        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
