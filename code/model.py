import torch
import torch.nn as nn
import torch.nn.functional as F

class PredOT(nn.Module):
    def __init__(self, n_word, dim, window, dropout, layer_output):
        super(PredOT, self).__init__()
        self.embed_word = nn.Embedding(n_word, dim)
        self.values_conv = nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window)
        self.weights_conv = nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim) for _ in range(layer_output)])
        self.activations = nn.ModuleList([nn.LeakyReLU() for _ in range(layer_output)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(layer_output)])
        self.W_pred = nn.Linear(2*dim, 1)
        
        
        self.dim = dim
        self.window = window
        self.layer_output = layer_output
        
    def forward(self, words):
        x = self.embed_word(words) # word2vec embedding
        x = x.transpose(1,2)
        values = self.values_conv(x)
        weights = self.weights_conv(x)
        weights = F.softmax(weights, dim=-1)
        xa = values * weights # attention weighted features
        xa_mean = torch.mean( xa, dim=-1) # Mean pooling
        xa_max, _ = torch.max( xa, dim=-1) # Max pooling
        pf = torch.cat([ xa_mean, xa_max ], dim=1) #Concat features for regression
        
        for j in range(self.layer_output):
            pf = self.W_out[j](pf)
            pf = self.activations[j](pf)
            pf = self.dropout_layers[j](pf)
            
        return self.W_pred(pf)
            
        
        
        
    
