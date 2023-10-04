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

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
    
    def forward(self, x):
        x = self.linear1(x)
        print(f"x after first linear layer: {x.size()}")
        x = self.relu(x)
        print(f"x after relu layer: {x.size()}")
        x = self.dropout(x)
        print(f"x after dropout layer: {x.size()}")
        x = self.linear2(x)
        print(f"x after 2nd linear layer: {x.size()}")
        return x 

class MultiHeadAttention():
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3*d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, sequence_length, d_model = x.size() # 30 x 200 x 512
        print(f"x.size(): {x.size()}")
        qkv = self.qkv_layer(x) # 30 x 200 x 1536
        print(f"qkv.size(): {qkv.size()}")
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        print(f"qkv afte   reshape.size(): {qkv.size()}")
        qkv = qkv.permute(0, 2, 1, 3) # 30 x 8 x 200 x 192
        print(f"qkv after permutation: {qkv.size()}")
        q, k, v = qkv.chunk(3, dim=-1) # 30 x 8 x 200 x 64
        print(f"q: {q.size()}, k: {k.size()}, v: {v.size()}")
        values, attention = scaled_dot_product(q, k, v, mask)
        print(f"values: {values.size()}, attention: {attention.size()}")
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        print(f"values after reshaping: {values.size()}")
        out = self.linear_layer(values)
        print(f"out after passing through linear layer: {out.size()}")
        return out







class MultiHeadCrossAttention():

class LayerNormalization(nn.Module):

    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):     
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        print(f"dims: {dims}")
        mean = inputs.mean(dim=dims, keepdim=True)


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model = d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, decoder_mask):
        _y = y
        print("MASKED SELF ATTENTION")
        y = self.self_attention(y, mask=decoder_mask)
        print("Drop Out 1")
        y = self.dropout1(y)
        print("Add + Layer Normalization 1")
        y = self.norm1(y + _y)
        
        _y = y 
        print("Cross Attention")

class SequentialDecoder():

class Decoder(nn.Module):


        