import torch.nn as nn


class TrasnformerEmbedding(nn.Module):
    
    def __init__(self, token_embed, pos_embed):
        super(TrasnformerEmbedding, self).__init__()
        self.embedding = nn.Sequential(token_embed, pos_embed)
        
        
    def forward(self, x):
        out = self.embedding(x)
        return out