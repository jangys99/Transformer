import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    '''
    code 자세히 살펴볼 필요가 있어보임
    제일 의문 -> exp을 이용해서 계산한 이유? => 수치적 안정성 때문.    
    '''
    def __init__(self, d_embed, max_len=256, device=torch.device('cpu')):
        super(PositionalEncoding, self).__init__()
        encoding = torch.zeros(max_len, d_embed)
        encoding.requires_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = encoding.unsqueeze(0).to(device)
        
        
    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoding[:, :seq_len, :]
        out = x + pos_embed
        return out
    
    
    
# a = torch.arange(0, 4, 2)
# print(a)

# b = -(math.log(10000.0) / 4)
# print(b)

# c = a * b
# print(c)
# print()

# y = torch.exp(c)
# print(y)


# position = torch.arange(0, 10).float().unsqueeze(1)
# print(position)

# encoding = torch.zeros(10, 4)
# print(encoding)

# encoding[:, 0::2] = torch.sin(position * y)
# encoding[:, 1::2] = torch.cos(position * y)

# print(encoding[:, 0::2])
# print(encoding[:, 1::2])

# print(encoding.shape)

# encoding = encoding.unsqueeze(0)
# print(encoding[:, :5, :].shape)