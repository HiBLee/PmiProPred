import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer import Encoder, EncoderLayer, Multi_Head_Self_Attention, PositionWiseFFN, PositionalEncoding, subsequent_mask

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class CNNBlock(nn.Module):
    def __init__(self, conv, in_channels, kernel_size, bias=True, bn=True, activator=nn.ReLU(True)):
        super(CNNBlock, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(in_channels, in_channels, kernel_size, bias=bias))
            if bn:
                modules_body.append(nn.BatchNorm1d(in_channels))
            if i == 0:
                modules_body.append(activator)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels, 1, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels, in_channels, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x) * x
        x = self.spatial_att(x) * x

        return x

# class FusionBlock(nn.Module):
#     def __init__(self, conv, in_channels, kernel_size, bias=True, bn=True, activator=nn.ReLU(True)):
#         super(FusionBlock, self).__init__()
#         modules_body = []
#         for i in range(2):
#             modules_body.append(conv(in_channels, in_channels, kernel_size, bias=bias))
#             if bn:
#                 modules_body.append(nn.BatchNorm1d(in_channels))
#             if i == 0:
#                 modules_body.append(activator)
#         self.body = nn.Sequential(*modules_body)
#
#     def forward(self, x):
#         res = self.body(x)
#         res += x
#         return res

class HTC(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, num_heads, n_layers, n_fusionblocks, ffn_num_hiddens, dropout):

        super(HTC, self).__init__()

        conv = default_conv

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.n_fusionblocks = n_fusionblocks
        self.ffn_num_hiddens = ffn_num_hiddens
        self.dropout = dropout

        self.mhsa = Multi_Head_Self_Attention(self.num_heads, self.embedding_dim, self.dropout)
        self.ffn = PositionWiseFFN(self.embedding_dim, self.ffn_num_hiddens, self.dropout)
        self.encoderlayer = EncoderLayer(self.embedding_dim, copy.deepcopy(self.mhsa), copy.deepcopy(self.ffn), self.dropout)

        self.Transformer_branch = nn.ModuleList([Encoder(self.encoderlayer, 2) for _ in range(self.n_layers)])
        self.CNN_branch_3 = nn.ModuleList([CNNBlock(conv, self.embedding_dim, 3) for _ in range(self.n_layers)])
        self.CNN_branch_5 = nn.ModuleList([CNNBlock(conv, self.embedding_dim, 5) for _ in range(self.n_layers)])
        self.CNN_branch_7 = nn.ModuleList([CNNBlock(conv, self.embedding_dim, 7) for _ in range(self.n_layers)])

        # self.fusion_block = nn.ModuleList([FusionBlock(conv, self.embedding_dim, 1) for _ in range(self.n_fusionblocks)])
        # self.fusion_block = nn.ModuleList([CBAM(4, 1) for _ in range(self.n_fusionblocks)])
        self.fusion_block = CBAM(4, 1)

    def forward(self, x):

        tb = PositionalEncoding(self.embedding_dim, self.dropout, 500)(x)
        mask = subsequent_mask(self.num_embeddings)

        cb3 = x.transpose(1, 2)
        cb5 = x.transpose(1, 2)
        cb7 = x.transpose(1, 2)

        for i in range(self.n_layers):
            tb = self.Transformer_branch[i](tb, mask.to(tb.device))
            cb3 = self.CNN_branch_3[i](cb3)
            cb5 = self.CNN_branch_5[i](cb5)
            cb7 = self.CNN_branch_7[i](cb7)

        tb = tb.unsqueeze(1)
        cb3 = cb3.transpose(1, 2).unsqueeze(1)
        cb5 = cb5.transpose(1, 2).unsqueeze(1)
        cb7 = cb7.transpose(1, 2).unsqueeze(1)

        feature_map_cat = torch.cat((tb, cb3, cb5, cb7), 1)
        x = self.fusion_block(feature_map_cat)
        # fb = feature_map_cat
        # for i in range(self.n_fusionblocks):
        #     fb = self.fusion_block[i](fb)
        # x = fb

        # f = tb + cb3 + cb5 + cb7
        # fb = f
        # fb = fb.transpose(1, 2)
        # for i in range(self.n_fusionblocks):
        #     fb = self.fusion_block[i](fb)
        # fb = fb.transpose(1, 2)
        # x = fb + f

        return x

class MLP(nn.Module):
    def __init__(self, num_embeddings, num_class, dropout=0.2):
        super(MLP, self).__init__()

        self.pool = nn.AdaptiveMaxPool2d((num_embeddings, 1))
        self.flat = nn.Flatten()
        self.dense1 = nn.Linear(4 * num_embeddings, num_embeddings)
        self.dense2 = nn.Linear(num_embeddings, num_class)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return F.softmax(self.dense2(self.dropout(F.relu(self.dense1(self.dropout(F.relu(self.flat(self.pool(x)))))))), dim=-1)

class miProPred_model(nn.Module):
    def __init__(self, token_size, embedding_dim, num_embeddings, num_heads, n_layers, n_fusionblocks, ffn_num_hiddens, dropout, num_class):
        super(miProPred_model, self).__init__()
        self.embeddings = nn.Embedding(token_size + 1, embedding_dim)
        self.htc = HTC(embedding_dim, num_embeddings, num_heads, n_layers, n_fusionblocks, ffn_num_hiddens, dropout)
        self.mlp = MLP(num_embeddings, num_class, dropout)
    def forward(self, x):
        return self.mlp(self.htc(self.embeddings(x)))

