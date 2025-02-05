#!/usr/bin/env python3
# 2020 Ruchao Fan

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PositionalEncoding(nn.Module):
    "Implement absolute position embedding."

    #Implementation similar to one in Vaswani et al. (add positional embedding directly to input)
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        #///OG Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class RelativePositionalEncoding(nn.Module):
    "Implement relative position embedding."
    def __init__(self, d_model, dropout, max_relative_len=10):
        super(RelativePositionalEncoding, self).__init__()

        #for range -max_relative_len - max_relative len precompute positional emb weights and store as embedding
        self.max_relative_len = max_relative_len
        max_len = 2 * max_relative_len + 1
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.embedding = nn.Embedding.from_pretrained(pe, freeze=True)

        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
    def forward(self, x):
        with torch.no_grad():
            distance = x.size(1) - 1
            #compute size of input vector, clamp input to rel_max len, add offset to remove 0 center and obtain embeddings
            range_vec = torch.arange(-distance, distance+1).type_as(x).long()
            index_vec = torch.clamp(range_vec, -self.max_relative_len, self.max_relative_len)
            index_vec = index_vec + self.max_relative_len
            pos_embed = self.embedding(index_vec)
        return (self.dropout(x), self.dropout(pos_embed))

        '''
        ///OG
        #range_mat = range_vec.repeat(x.size(1)).view(x.size(1), x.size(1))
        #index_mat = range_mat.transpose(0, 1) - range_mat
        #index_mat_clipped = torch.clamp(index_mat, -self.max_relative_len, self.max_relative_len)
        #final_mat = index_mat_clipped + self.max_relative_len
        #return self.embedding(final_mat).view(x.size(1), x.size(1), -1)
        '''

class TextEmbedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(TextEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class Padding(nn.Module):
    def __init__(self, pad=(1,1,1,1)):
        super(Padding, self).__init__()
        self.pad = pad

    def forward(self, x):
        pad_x = F.pad(x, self.pad, "constant", 0)
        return pad_x

class ConvEmbedding(nn.Module):
    """
    Mapping input features to embeddings and downsample with 4.
    """
    def __init__(self, input_size, d_model, dropout, pos_enc, causal=False):
        super(ConvEmbedding, self).__init__()

        if causal:
            conv1 = nn.Conv2d(1, d_model, (2,3), (2,2), (0,1))
            conv2 = nn.Conv2d(d_model, d_model, (2,3), (2,2), (0,1))
            self.conv = nn.Sequential(Padding(pad=(0, 0, 1, 0)), conv1, nn.ReLU(),
                                      Padding(pad=(0, 0, 1, 0)), conv2, nn.ReLU())
        else:
            conv1 = nn.Conv2d(1, d_model, 3, 2, 1)
            conv2 = nn.Conv2d(d_model, d_model, 3, 2, 1)
            self.conv = nn.Sequential(conv1, nn.ReLU(), conv2, nn.ReLU())

        self.d_model = d_model 
        
        self.linear_out = nn.Linear(d_model * (((input_size-1)//2) // 2 + 1), d_model)
        self.pos_enc = pos_enc
        
    #pass input through cnn, then extract pos_emb using emb layer; also downsample mask to match dimensions of x
    def forward(self, x, mask):
        "mask needs to be revised to downsample version"
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, d = x.size()
        x = self.linear_out(x.transpose(1,2).contiguous().view(b, t, c*d))
        
        x = self.pos_enc(x * math.sqrt(self.d_model))
        if mask.size(1) == 1:
            mask = mask[:, :, ::2][:, :, ::2]
        else:
            mask = mask[:,::2,::2][:, ::2,::2]
        return x, mask


