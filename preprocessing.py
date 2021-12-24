#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess IWSLT14 En-Zh Dataset, implement DataSet
Thanks to Harvard Annoated Transformer in http://nlp.seas.harvard.edu/2018/04/03/attention.html

@author: Ma (Ma787639046@outlook.com)
@data: 2020/12/24

"""
import logging
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
import config

def chinese_tokenizer_load():
    sp_ch = spm.SentencePieceProcessor()
    sp_ch.Load('./data/ch.model')
    logging.info(f'SentencePiece loaded at ./data/ch.model')
    return sp_ch

def english_tokenizer_load():
    sp_en = spm.SentencePieceProcessor()
    sp_en.Load('./data/en.model')
    logging.info(f'SentencePiece loaded at ./data/en.model')
    return sp_en

def subsequent_mask(size):  # 生成下三角布尔矩阵
    """Mask out subsequent positions."""
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)
    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    """Object for holding a batch of data with mask during training."""
    def __init__(self, src_text, trg_text, src, trg=None, pad=0):
        self.src_text = src_text
        self.trg_text = trg_text
        self.src = src
        # 对于当前输入的句子非空部分进行判断成bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.src_mask = (src != pad).unsqueeze(-2)
        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        if trg is not None:
            # decoder要用到的target输入部分
            self.trg = trg[:, :-1]
            # decoder训练时应预测输出的target结果
            self.trg_y = trg[:, 1:]
            # 将target输入部分进行attention mask
            self.trg_mask = self.make_std_mask(self.trg, pad)
            # 将应输出的target结果中实际的词数进行统计
            self.ntokens = (self.trg_y != pad).data.sum()

    # Mask掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MTDataset(Dataset):
    def __init__(self, ch_data_path, en_data_path, rank=None):
        self.en_sent, self.cn_sent = self.get_dataset(ch_data_path, en_data_path, sort=True)   # 原本sort=True
        self.sp_en = english_tokenizer_load()
        self.sp_ch = chinese_tokenizer_load()
        self.PAD = self.sp_en.pad_id()  # 0
        self.BOS = self.sp_en.bos_id()  # 2
        self.EOS = self.sp_en.eos_id()  # 3
        self.rank = rank

    @staticmethod
    def len_argsort(seq): # 传入句子列表(分好词的二维列表)，按照句子长度排序后，返回排序后原来各句子在数据中的索引下标
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, ch_data_path, en_data_path, sort=False):
        with open(ch_data_path, 'r') as f:
            cn_sent = f.readlines()
        with open(en_data_path, 'r') as f:
            en_sent = f.readlines()
        assert len(cn_sent) == len(en_sent), f"Number of lines in {ch_data_path} must be equal to {en_data_path}"
        """
            Sort 排序:
            排序数据可以使一个batch内的数据长度尽量相似。
            以英文句子长度排序的(句子下标)顺序为基准，把中文和英文按照同样的顺序排序。
        """
        if sort:
            sorted_index = self.len_argsort(en_sent)
            en_sent = [en_sent[i] for i in sorted_index]
            cn_sent = [cn_sent[i] for i in sorted_index]
        return en_sent, cn_sent

    def __getitem__(self, idx):
        en_text = self.en_sent[idx]
        cn_text = self.cn_sent[idx]
        return [en_text, cn_text]

    def __len__(self):
        return len(self.en_sent)

    def collate_fn(self, batch):
        src_text = [x[0] for x in batch]    # en_text
        tgt_text = [x[1] for x in batch]    # cn_text

        # Tokenize with SentencePiece, add [BOS] & [EOS]
        src_tokens = [[self.BOS] + self.sp_en.EncodeAsIds(sent) + [self.EOS] for sent in src_text]
        tgt_tokens = [[self.BOS] + self.sp_ch.EncodeAsIds(sent) + [self.EOS] for sent in tgt_text]

        # Pad Sequence
        src_pad = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in src_tokens],
                                   batch_first=True, padding_value=self.PAD)
        tgt_pad = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in tgt_tokens],
                                    batch_first=True, padding_value=self.PAD)
        if self.rank != None:
            src_pad = src_pad.to(self.rank)
            tgt_pad = tgt_pad.to(self.rank)

        return Batch(src_text, tgt_text, src_pad, tgt_pad, self.PAD)
