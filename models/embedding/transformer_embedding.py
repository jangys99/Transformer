import torch.nn as nn


class TransformerEmbedding(nn.Module):
    '''
    Dropout 적용
    '''
    def __init__(self, token_embed, pos_embed, dr_rate=0):
        super(TransformerEmbedding, self).__init__()
        self.embedding = nn.Sequential(token_embed, pos_embed)
        self.dropout = nn.Dropout(p=dr_rate)
        
        
    def forward(self, x):
        out = self.embedding(x)
        out = self.dropout(out)
        return out