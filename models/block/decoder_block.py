import copy
import torch.nn as nn

from models.layer.residual_connection_layer import ResidualConnectionLayer


class DecoderBlock(nn.Module):
    '''
    layer normalization, Dropout 적용
    '''
    def __init__(self, self_attention, cross_attention, position_ff, norm, dr_rate=0):
        super(DecoderBlock, self).__init__()
        self.self_attention = self_attention
        self.residuals1 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.cross_attention = cross_attention
        self.residuals2 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        self.position_ff = position_ff
        self.residuals3 = ResidualConnectionLayer(copy.deepcopy(norm), dr_rate)
        # self.residuals = [ResidualConnectionLayer() for _ in range(3)]
        
        
    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        out = self.residuals1(out, lambda out: self.self_attention(query=out, key=out, value=out, mask=tgt_mask))
        out = self.residuals2(out, lambda out: self.cross_attention(query=out, key=encoder_out, value=encoder_out, mask= src_tgt_mask))
        out = self.residuals3(out, self.position_ff)
        return out