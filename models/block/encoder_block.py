import copy
import torch.nn as nn

from models.layer.residual_connection_layer import ResidualConnectionLayer


class EncoderBlock(nn.Module):
    '''
    layer normalization, Dropout 적용
    '''
    def __init__(self, self_attention, position_ff, norm, dr_rate=0):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.residuals1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)        
        self.position_ff = position_ff
        self.residuals2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        #self.residuals = [ResidualConnectionLayer() for _ in range(2)]
        
    def forward(self, src, src_mask):
        out = src
        out = self.residuals1(out, lambda out: self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residuals2(out, self.position_ff)
        return out