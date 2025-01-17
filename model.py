#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model implement of Machine Translation task using Transformer on IWSLT14 En-Zh Dataset
Thanks to Harvard Annoated Transformer in http://nlp.seas.harvard.edu/2018/04/03/attention.html

@author: Ma (Ma787639046@outlook.com)
@data: 2020/12/24

"""
import torch.nn as nn
from transformers_utils import copy, MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward, Embeddings, Transformer, Encoder, EncoderLayer, Decoder, DecoderLayer, Generator

def transformer_encoder_decoder_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    # 实例化Attention对象
    attn = MultiHeadedAttention(h, d_model)
    # 实例化FeedForward对象
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 实例化PositionalEncoding对象
    position = PositionalEncoding(d_model, dropout)
    # 实例化Transformer模型对象
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  # xavier_uniform init
    return model

