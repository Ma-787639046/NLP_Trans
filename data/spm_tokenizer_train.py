#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train English & Chinese SentencePiece models on IWSLT14 En-Zh Dataset

SentencePiece is used in this en-zh Translation Project.
Generate corpus.en & corpus.zh SentencePiece models for Tokenizing Usage
See: https://github.com/google/sentencepiece/tree/master/python

@author: Ma (Ma787639046@outlook.com)
@data: 2020/12/23

"""

import os
import sentencepiece as spm

def train(input_file, vocab_size, model_prefix, model_type, character_coverage):
    """
    search on https://github.com/google/sentencepiece/blob/master/doc/options.md to learn more about the parameters
    :param input_file: one-sentence-per-line raw corpus file. No need to run tokenizer, normalizer or preprocessor.
                       By default, SentencePiece normalizes the input with Unicode NFKC.
                       You can pass a comma-separated list of files.
    :param vocab_size: vocabulary size, e.g., 8000, 16000, or 32000
    :param model_prefix: output model name prefix. <model_prefix>.model and <model_prefix>.vocab are generated.
    :param model_type: model type. Choose from unigram (default), bpe, char, or word.
                       The input sentence must be pretokenized when using word type.
    :param character_coverage: amount of characters covered by the model, good defaults are: 0.9995 for languages with
                               rich character set like Japanse or Chinese and 1.0 for other languages with
                               small character set.
    """
    
    os.chdir(os.path.abspath('.'))
    args =  f'--input={input_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --model_type={model_type} '\
            f'--character_coverage={character_coverage} --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'
    spm.SentencePieceTrainer.Train(args)


def run():
    en_input = './corpus/test.en,./corpus/train.en,./corpus/valid.en'
    en_vocab_size = 32000
    en_model_prefix = 'en'
    en_model_type = 'bpe'
    en_character_coverage = 1
    train(en_input, en_vocab_size, en_model_prefix, en_model_type, en_character_coverage)

    ch_input = './corpus/test.zh,./corpus/train.zh,./corpus/valid.zh'
    ch_vocab_size = 32000
    ch_model_prefix = 'ch'
    ch_model_type = 'bpe'
    ch_character_coverage = 0.9995
    train(ch_input, ch_vocab_size, ch_model_prefix, ch_model_type, ch_character_coverage)


def test():
    sp = spm.SentencePieceProcessor()
    text = "美国总统特朗普今日抵达夏威夷。"

    sp.Load("./ch.model")
    print(sp.EncodeAsPieces(text))
    print(sp.EncodeAsIds(text))
    a = [12907, 277, 7419, 7318, 18384, 28724]
    print(sp.decode_ids(a))


if __name__ == "__main__":
    run()
    # test()