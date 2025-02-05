#!/usr/bin/env python3
# 2020 Ruchao Fan
# 2022 Ruchao Fan

import torch.nn as nn
from models.modules.norm import LayerNorm
from models.modules.utils import clones, SublayerConnection

class SelfAttLayer(nn.Module):
    "Attention block with self-attn and feed forward (defined below)"

    #refer to comments on fanat_conformer_blocks
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(SelfAttLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, self_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, self_mask))
        return self.sublayer[1](x, self.feed_forward)

class SrcAttLayer(nn.Module):
    "Attention block with src-attn and feed forward"

    #refer to comments on fanat_conformer_blocks
    def __init__(self, size, src_attn, feed_forward, dropout):
        super(SrcAttLayer, self).__init__()
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    
    def forward(self, x, m, src_mask):
        x = self.sublayer[0](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[1](x, self.feed_forward)
        
class MixAttLayer(nn.Module):
    "Attention block with self-attn, src-attn, and feed forward (defined below)"

    #refer to comments on fanat_conformer_blocks
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(MixAttLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, self_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, self_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Mix3AttLayer(nn.Module):
    "Attention block with self-attn, src-attn from audio, src-attn from text, and feed forward (defined below)"

    #refer to comments on fanat_conformer_blocks
    def __init__(self, size, self_attn, src_attn_audio, src_attn_text, feed_forward, dropout):
        super(Mix3AttLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn_audio = src_attn_audio
        self.src_attn_text = src_attn_text
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 4)
 
    def forward(self, x, memory_audio, memory_text, src_mask_audio, src_mask_text, self_mask):
        ma, mt = memory_audio, memory_text
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, self_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn_audio(x, ma, ma, src_mask_audio))
        x = self.sublayer[2](x, lambda x: self.src_atn_text(x, mt, mt, src_mask_text))
        return self.sublayer[3](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    #refer to comments on fanat_conformer_blocks
    def __init__(self, size, self_attn, feed_forward, dropout, N):
        super(Encoder, self).__init__()
        layer = SelfAttLayer(size, self_attn, feed_forward, dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask, interctc_alpha=0, interctc_layer=6):
        "Pass the input (and mask) through each layer in turn."
        n_layer = 0
        for layer in self.layers:
            x = layer(x, mask)
            if interctc_alpha > 0 and n_layer == interctc_layer - 1:
                inter_out = x
            n_layer += 1

        if interctc_alpha > 0:
            return (self.norm(x), inter_out)
        else:
            return self.norm(x)

class AcEmbedExtractor(nn.Module):

    #refer to comments on fanat_conformer_blocks
    "Extractor sub-unit level acoustic embedding with CTC segments"
    def __init__(self, size, src_attn, feed_forward, dropout, N):
        super(AcEmbedExtractor, self).__init__()
        layer = SrcAttLayer(size, src_attn, feed_forward, dropout)
        self.layers = clones(layer, N)

    def forward(self, x, memory, trigger_mask):
        for layer in self.layers:
            x = layer(x, memory, trigger_mask)
        return x

class SelfAttDecoder(nn.Module):
    "Token acoustic embedding with self attention layers, similar with Encoder structure"

    #refer to comments on fanat_conformer_blocks
    def __init__(self, size, self_attn, feed_forward, dropout, N):
        super(SelfAttDecoder, self).__init__()
        layer = SelfAttLayer(size, self_attn, feed_forward, dropout)
        self.layers = clones(layer, N)
        
    def forward(self, x, mask, interce_alpha=0, interce_layer=4):
        "Pass the input (and mask) through each layer in turn."
        n_layer = 0
        for layer in self.layers:
            x = layer(x, mask)
            if interce_alpha > 0 and n_layer == interce_layer - 1:
                interce_out = x
            n_layer += 1

        if interce_alpha > 0:
            return (x, interce_out)
        else:
            return x

class MixAttDecoder(nn.Module):
    "Generic N layer decoder with masking."

    #refer to comments on fanat_conformer_blocks
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout, N):
        super(MixAttDecoder, self).__init__()
        layer = MixAttLayer(size, self_attn, src_attn, feed_forward, dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask, interce_alpha=0, interce_layer=4):
        n_layer = 0
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
            if interce_alpha > 0 and n_layer == interce_layer - 1:
                interce_out = x
            n_layer += 1

        if interce_alpha > 0:
            return (self.norm(x), interce_out)
        else:
            return self.norm(x)

class Mix3AttDecoder(nn.Module):
    "Generic N layer decoder with masking."

    #refer to comments on fanat_conformer_blocks
    def __init__(self, size, self_attn, src_attn_audio, src_attn_text, feed_forward, dropout, N):
        super(Mix3AttDecoder, self).__init__()
        layer = MixAttLayer(size, self_attn, src_attn_audio, src_attn_text, feed_forward, dropout)
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory_audio, memory_text, src_mask_audio, src_mask_text, tgt_mask, interce_alpha=0, interce_layer=4):
        n_layer = 0
        for layer in self.layers:
            x = layer(x, memory_audio, memory_text, src_mask_audio, src_mask_text, tgt_mask)
            if interce_alpha > 0 and n_layer == interce_layer - 1:
                interce_out = x
            n_layer += 1

        if interce_alpha > 0:
            return (self.norm(x), interce_out)
        else:
            return self.norm(x)



