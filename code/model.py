import torch
import esm
import torch.nn as nn
import torch.nn.functional as F

class PredOT(nn.Module):
    def __init__(self, device, window, dropout, layer_cnn, layer_output):
        super(PredOT, self).__init__()
        dim = 320
        self.device = device
        self.dim = dim
        self.layer_output = layer_output
        self.layer_cnn = layer_cnn
        
        self.esm2_model, self.esm2_batch_converter = self.load_ESM2_model()
        self.Vconvs = nn.ModuleList([ nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window) for _ in range(layer_cnn)])
        self.Wconvs = nn.ModuleList([ nn.Conv1d(dim, dim, kernel_size=2*window+1, padding=window) for _ in range(layer_cnn)])    
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim) for _ in range(layer_output)])
        self.W_pred = nn.Linear(2*dim, 1)
        
        
    def load_ESM2_model(self):
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D() # 8M params, 6 layers
        model = model.half()#float16
        model = model.to(self.device)
        batch_converter = alphabet.get_batch_converter()
        return model, batch_converter
    
    def get_ESM2_embeddings(self, ids, seqs):
        data = [(ids[i], seqs[i]) for i in range(len(ids))]
        batch_labels, batch_strs, batch_tokens = self.esm2_batch_converter(data)
        batch_tokens = batch_tokens.half()#float16
        batch_tokens = batch_tokens.to(device=self.device, non_blocking=True)
        with torch.no_grad():
            emb = self.esm2_model(batch_tokens, repr_layers=[6], return_contacts=False)
        emb = emb["representations"][6]
        emb = emb.transpose(1,2) # (batch, features, seqlen)
        emb = emb.to(self.device)
        return emb
    
    def get_values(self,x):
        for i in range(self.layer_cnn):
            x = F.leaky_relu( self.Vconvs[i](x) )
        return x
    
    
    def get_weights(self,x):
        for i in range(self.layer_cnn):
            x = F.leaky_relu( self.Wconvs[i](x) )
        return x
    
    
    def forward(self, ids, seqs):
        emb = self.get_ESM2_embeddings(ids, seqs)
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
            
        
        
        
    
