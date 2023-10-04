import torch
import math
from torch import nn
import torch.nn.functional as F

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    print(f"scaled.size() : {scaled.size()}")
    if mask is not None:
        print(f"-- Adding Mask of Shape {mask.size()} --")
        scaled += mask 
    attention = F.softmax(scaled, dim = 1)
    values = torch.matmul(attention, v)
    return values, attention
