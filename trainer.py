#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train function implementation
Thanks to Harvard Annoated Transformer in http://nlp.seas.harvard.edu/2018/04/03/attention.html

@author: Ma (Ma787639046@outlook.com)
@data: 2020/12/24

"""
import torch
import logging
import sacrebleu
from tqdm import tqdm

import config
from preprocessing import chinese_tokenizer_load
from decode import beam_search

def run_epoch(dataloader, model, criterion, optimizer=None, scheduler=None):
    """进行一个epoch的训练"""
    total_tokens = 0.
    total_loss = 0.

    for batch in tqdm(dataloader):
        out = model(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = criterion(out.contiguous().view(-1, out.size(-1)),
                              batch.trg_y.contiguous().view(-1)) / batch.ntokens
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        total_loss += loss.data.item() * batch.ntokens.float()
        total_tokens += batch.ntokens
    return total_loss / total_tokens


def train(train_dataloader, dev_dataloader, model, criterion, optimizer, scheduler, global_rank):
    """训练并保存模型"""
    if global_rank == 0:
        logging.info("------ Start Training! ------")
    best_bleu_score = 0.0
    early_stop = config.early_stop
    for epoch in range(1, config.epoch_num + 1):
        # 模型训练
        if global_rank == 0:
            logging.info(f"[Epoch {epoch}] Trainging...")
        model.train()
        train_loss = run_epoch(train_dataloader, model, criterion, optimizer, scheduler)
        # 计算bleu分数
        if global_rank == 0:
            logging.info(f"[Epoch {epoch}] Validating...")
            model.eval()
            bleu_score = evaluate(dev_dataloader, model)
            logging.info(f'Epoch: {epoch:2d}, loss: {train_loss:.3f}, Bleu Score: {bleu_score}')

            # 基于bleu分数，设置early stop
            if bleu_score > best_bleu_score:
                torch.save(model.state_dict(), config.model_path)
                best_bleu_score = bleu_score
                early_stop = config.early_stop
                logging.info("-------- Save Best Model! --------")
            else:
                early_stop -= 1
                logging.info("Early Stop Left: {}".format(early_stop))
            if early_stop == 0:
                logging.info("-------- Early Stop! --------")
                break

def evaluate(data, model, mode='dev'):
    """在data上用训练好的模型进行预测，打印模型翻译结果"""
    sp_chn = chinese_tokenizer_load()
    src = []
    trg = []
    res = []
    with torch.no_grad(): 
        for batch in tqdm(data):
            cn_sent = batch.trg_text    # Chinese text
            en_text = batch.src_text    # English text
            src_mask = (batch.src != 0).unsqueeze(-2)
            decode_result, _ = beam_search(model.module, batch.src, src_mask, config.max_len,
                                            config.padding_idx, config.bos_idx, config.eos_idx,
                                            config.beam_size, config.device)
            decode_result = [h[0] for h in decode_result]
            translation = [sp_chn.decode_ids(_s) for _s in decode_result]
            src.append(en_text)
            trg.extend(cn_sent)
            res.extend(translation)
    if mode == 'test':  # 如果是test集，输出“参考中文trg”和“模型预测的中文res”
        with open(config.output_path, "w") as fp:
            for i in range(len(trg)):
                fp.write(f"idx: {i}\n")
                fp.write(f"English sentence: {src[i]}\n")
                fp.write(f"Translation: {res[i]}\n")
                fp.write(f"References:  {trg[i]}\n")
    trg = [trg]
    bleu = sacrebleu.corpus_bleu(res, trg, tokenize='zh')
    return float(bleu.score)


# def test(data, model):
#     with torch.no_grad():
#         # 加载模型
#         model.load_state_dict(torch.load(config.model_path))
#         model_par = torch.nn.DataParallel(model)
#         model.eval()
#         # 开始预测
#         bleu_score = evaluate(data, model, 'test')
#         logging.info(f'Test Bleu Score: {bleu_score}')


# def translate(src, model, use_beam=True):
#     """用训练好的模型进行预测单句，打印模型翻译结果"""
#     sp_chn = chinese_tokenizer_load()
#     with torch.no_grad():
#         model.load_state_dict(torch.load(config.model_path))
#         model.eval()
#         src_mask = (src != 0).unsqueeze(-2)
#         if use_beam:
#             decode_result, _ = beam_search(model, src, src_mask, config.max_len,
#                                            config.padding_idx, config.bos_idx, config.eos_idx,
#                                            config.beam_size, config.device)
#             decode_result = [h[0] for h in decode_result]
#         else:
#             decode_result = batch_greedy_decode(model, src, src_mask, max_len=config.max_len)
#         translation = [sp_chn.decode_ids(_s) for _s in decode_result]
#         print(translation[0])