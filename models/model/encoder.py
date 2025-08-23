import copy
import torch.nn as nn

class Encoder(nn.Module):
    '''
    dropout 적용
    '''
    def __init__(self, encoder_block, n_layer, norm): # n_layers : encoder block 개수
        super(Encoder, self).__init__()
        self.n_layer = n_layer
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(self.n_layer)])
        self.norm = norm
            
    
    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)    # 이전 block의 output을 이후 block의 input으로 사용
        out = self.norm(out)
        return out