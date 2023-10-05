import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]

class PositionalEncoding():

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