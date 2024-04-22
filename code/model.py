import torch
import torch.nn as nn
import torch.nn.functional as F

class PredOT(nn.Module):
    def __init__(self, n_word, dim, window, dropout, layer_cnn, layer_output):
        super(PredOT, self).__init__()
        self.embed_word = nn.Embedding(n_word, dim)
        self.Vconvs = nn.ModuleList([ nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window) for _ in range(layer_cnn)])
        self.Wconvs = nn.ModuleList([ nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window) for _ in range(layer_cnn)])    
#         self.values_conv = nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window)
#         self.weights_conv = nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim) for _ in range(layer_output)])
        self.W_pred = nn.Linear(2*dim, 1)
        
        self.dim = dim
        self.layer_output = layer_output
        self.layer_cnn = layer_cnn
        
        
    def get_values(self,x):
        for i in range(self.layer_cnn):
            x = self.Vconvs[i](x)
        return x
    
    
    def get_weights(self,x):
        for i in range(self.layer_cnn):
            x = self.Wconvs[i](x)
        return x
    
    
    def forward(self, words):
        x = self.embed_word(words) # Embedding
        x = x.transpose(1,2)  
        values = self.get_values(x) # CNN
        weights = self.get_weights(x) # CNN
        weights = F.softmax(weights, dim=-1)
        xa = values * weights # Attention weighted features
        xa_mean = torch.mean( xa, dim=-1) # Mean pooling
        xa_max, _ = torch.max( xa, dim=-1) # Max pooling
        
        y = torch.cat([ xa_mean, xa_max ], dim=1) #Concat features for regression
        
        for j in range(self.layer_output):
            y =  F.leaky_relu( self.W_out[j](y) )
            
        return self.W_pred(y)
            
        
        
        
    
