import copy
import torch.nn as nn

class Encoder(nn.Module):
    
    def __init__(self, encoder_block, n_layer): # n_layers : encoder block 개수
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_block))
            
    
    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)    # 이전 block의 output을 이후 block의 input으로 사용
        return out