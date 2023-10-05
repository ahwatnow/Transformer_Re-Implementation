import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute + mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_sequence_length):
            super().__init__()
            self.max_sequence_length = max_sequence_length
            self.d_model = d_model
    
    def forward(self):
        

class SentenceEmbedding():

class MultiHeadAttention():

class LayerNormalization():

class PositionWiseFeedForward():

class EncoderLayer():

class SequentialEncoder():

class Encoder():

class MultiHeadCrossAttention(): 

class DecoderLayer():

class SequentialDecoder():

class Decoder(): 

class Transformer():