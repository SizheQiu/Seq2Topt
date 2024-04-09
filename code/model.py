import torch
import torch.nn as nn
import torch.nn.functional as F

class PredOT(nn.Module):
    def __init__(self, n_word, dim, window, dropout, layer_output):
        super(PredOT, self).__init__()
        self.embed_word = nn.Embedding(n_word, dim)
        self.values_conv = nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window)
        self.weights_conv = nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window)
        
        self.W_out = nn.ModuleList([nn.Linear(3*dim, 3*dim) for _ in range(layer_output)])
        self.activations = nn.ModuleList([nn.LeakyReLU() for _ in range(layer_output)])
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(layer_output)])
        self.W_pred = nn.Linear(3*dim, 1)
        
        self.dim = dim
        self.layer_output = layer_output
        
    def forward(self, words):
        x = self.embed_word(words) # word2vec embedding
        x = x.transpose(1,2)
        values = self.values_conv(x)
        weights = self.weights_conv(x)
        weights = F.softmax(weights, dim=-1)
        xa = values * weights # Attention weighted features
        xa_mean = torch.mean( xa, dim=-1) # Mean pooling
        xa_max, _ = torch.max( xa, dim=-1) # Max pooling
        xa_min,_ = torch.min(xa, dim=-1) # Min pooling
        
        y = torch.cat([ xa_mean, xa_max, xa_min ], dim=1) #Concat features for regression
        
        for j in range(self.layer_output):
            y = self.W_out[j](y)
            y = self.activations[j](y)
            y = self.dropout_layers[j](y)
            
        return self.W_pred(y)
            
        
        
        
    
