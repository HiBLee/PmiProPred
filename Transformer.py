import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_dim)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * -(math.log(10000.0) / embedding_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        device = x.device
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False).to(device)
        return self.dropout(x)


def subsequent_mask(size):
    atten_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(atten_shape), k=1).astype('uint8')
    return torch.from_numpy(1 - subsequent_mask)


def attention(q, k, v, mask=None, dropout=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    atten = F.softmax(scores, dim=-1)
    if dropout is not None:
        atten = dropout(atten)

    return torch.matmul(atten, v), atten


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Multi_Head_Self_Attention(nn.Module):
    def __init__(self, num_heads, embedding_dim, dropout=0.2):
        super(Multi_Head_Self_Attention, self).__init__()
        assert embedding_dim % num_heads == 0
        self.d_k = embedding_dim // num_heads
        self.num_heads = num_heads
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        self.atten = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = q.size(0)
        q, k, v = [model(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) for model, x in zip(self.linears, (q, k, v))]
        x, self.atten = attention(q, k, v, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        return self.linears[-1](x)


class PositionWiseFFN(nn.Module):
    def __init__(self, embedding_dim, ffn_num_hiddens, dropout=0.2):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(embedding_dim, ffn_num_hiddens)
        self.dense2 = nn.Linear(ffn_num_hiddens, embedding_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dense2(self.dropout(F.relu(self.dense1(x))))


class SubLayerConnection(nn.Module):
    def __init__(self, embedding_dim, dropout=0.2):
        super(SubLayerConnection, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(self.norm(x))))


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, self_atten, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.self_atten = self_atten
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(p=dropout)
        self.sublayer = clones(SubLayerConnection(embedding_dim, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_atten(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.embedding_dim)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

