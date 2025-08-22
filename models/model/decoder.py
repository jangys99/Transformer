import copy
import torch.nn as nn


class Decoder(nn.Module):
    '''
    tgt_mask : Decoder input으로 주어지는 target sentence의 pad masking과 subsequent masking (make_tgt_mask())
    src_tgt_mask : Self-Multi-Head Attention layer에서 넘어온 query, Encoder에서 넘어온 key, value 사이의 pad masking.
    '''
    def __init__(self, decoder_block, n_layer):
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(self.n_layer)])
        
        
    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        for layer in self.layers:
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
        return out