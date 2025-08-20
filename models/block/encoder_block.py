import torch.nn as nn


class EncoderBlock(nn.Module):
    
    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = self.position_ff
        
        
    def forward(self, x):
        out = x
        out = self.self_attention(out)
        out = self.position_ff(out)
        return out