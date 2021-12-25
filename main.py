#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Translation Task using Transformer: English -> Chinese
Thanks to Harvard Annoated Transformer in http://nlp.seas.harvard.edu/2018/04/03/attention.html

@author: Ma (Ma787639046@outlook.com)
@data: 2020/12/24

"""
import os
import utils
import config
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers.optimization import get_polynomial_decay_schedule_with_warmup
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from trainer import train, test
from preprocessing import MTDataset, english_tokenizer_load
from model import transformer_encoder_decoder_model

def run(rank):
    utils.set_logger(config.log_path)
    world_size = config.n_gpu * config.n_node
    global_rank = config.node_rank * config.n_gpu + rank
    if rank == 0:
        if not os.path.exists(config.temp_dir):
            os.makedirs(config.temp_dir)
    # preparing the distributed env
    # using nccl for distributed training
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=global_rank)

    train_dataset = MTDataset(ch_data_path=config.train_ch_data_path, en_data_path=config.train_en_data_path, rank=rank)
    dev_dataset = MTDataset(ch_data_path=config.dev_ch_data_path, en_data_path=config.dev_en_data_path, rank=rank)
    test_dataset = MTDataset(ch_data_path=config.test_ch_data_path, en_data_path=config.test_en_data_path, rank=rank)
    if rank == 0: 
        logging.info("-------- Dataset Build! --------")

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn, sampler=sampler)
    # dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
    #                             collate_fn=dev_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn, sampler=sampler)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn, sampler=sampler)
    if rank == 0: 
        logging.info("-------- Get Dataloader! --------")

    # 初始化模型
    model = transformer_encoder_decoder_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    if torch.cuda.is_available():    # Move model to GPU:rank
        model.cuda(rank)

    if rank == 0: 
        logging.info("finish Loading model")
    
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=False
    )

    dist.barrier()  # synchronizes all processes
    if rank == 0:
        logging.info("Finish initing all processors.")

    # 训练
    total_steps = 1.0 * len(train_dataloader) * config.epoch_num
    warmup_steps = config.warmup_proportion * total_steps
    logging.info(f"Scheduler: total_steps:{total_steps}, warmup_steps:{warmup_steps}")

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps
    )

    train(train_dataloader, dev_dataloader, model, criterion, optimizer, scheduler, rank)
    # test(test_dataloader, model, rank)
    dist.destroy_process_group()


# def one_sentence_translate(sent, beam_search=True):
#     # 初始化模型
#     model = transformer_encoder_decoder_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
#                        config.d_model, config.d_ff, config.n_heads, config.dropout)
#     BOS = english_tokenizer_load().bos_id()  # 2
#     EOS = english_tokenizer_load().eos_id()  # 3
#     src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
#     batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
#     translate(batch_input, model, use_beam=beam_search)


# def translate_example():
#     """单句翻译示例"""
#     sent = "The near-term policy remedies are clear: raise the minimum wage to a level that will keep a " \
#            "fully employed worker and his or her family out of poverty, and extend the earned-income tax credit " \
#            "to childless workers."
#     # tgt: 近期的政策对策很明确：把最低工资提升到足以一个全职工人及其家庭免于贫困的水平，扩大对无子女劳动者的工资所得税减免。
#     one_sentence_translate(sent, beam_search=True)


if __name__ == "__main__":
    # import warnings
    # warnings.filterwarnings('ignore')

    os.environ['MASTER_ADDR'] = config.MASTER_ADDR
    os.environ['MASTER_PORT'] = config.MASTER_PORT

    mp.spawn(run, nprocs=config.n_gpu)
    # translate_example()
