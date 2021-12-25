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
import logging
from argparse import ArgumentParser
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

def run(rank, *tuples):
    args = tuples[0]    # mp.spawm() pass a tuple (args, ) to train() function, so args = tuples[0]
    world_size = args.n_gpu * args.n_node
    global_rank = args.node_rank * args.n_gpu + rank
    if global_rank == 0:
        utils.set_logger(args.log_path)
        if not os.path.exists(args.temp_dir):
            os.makedirs(args.temp_dir)
    # preparing the distributed env
    # using nccl for distributed training
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=global_rank)

    train_dataset = MTDataset(ch_data_path=args.train_ch_data_path, en_data_path=args.train_en_data_path, rank=rank)
    dev_dataset = MTDataset(ch_data_path=args.dev_ch_data_path, en_data_path=args.dev_en_data_path, rank=rank)
    test_dataset = MTDataset(ch_data_path=args.test_ch_data_path, en_data_path=args.test_en_data_path, rank=rank)
    if rank == 0: 
        logging.info(f"-------- Dataset Build! --------")

    sampler_train = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank)
    sampler_dev = DistributedSampler(dev_dataset, num_replicas=world_size, rank=global_rank)
    sampler_test = DistributedSampler(test_dataset, num_replicas=world_size, rank=global_rank)

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn, sampler=sampler_train)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn, sampler=sampler_dev)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                                 collate_fn=test_dataset.collate_fn, sampler=sampler_test)
    if rank == 0: 
        logging.info("-------- Get Dataloader! --------")

    # 初始化模型
    model = transformer_encoder_decoder_model(args.src_vocab_size, args.tgt_vocab_size, args.n_layers,
                       args.d_model, args.d_ff, args.n_heads, args.dropout)
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
    total_steps = 1.0 * len(train_dataloader) * args.epoch_num
    warmup_steps = args.warmup_proportion * total_steps
    logging.info(f"Scheduler: total_steps:{total_steps}, warmup_steps:{warmup_steps}")

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps
    )

    train(train_dataloader, dev_dataloader, model, criterion, optimizer, scheduler, rank, args)
    # test(test_dataloader, model, rank, args)
    dist.destroy_process_group()


# def one_sentence_translate(sent, beam_search=True):
#     # 初始化模型
#     model = transformer_encoder_decoder_model(args.src_vocab_size, args.tgt_vocab_size, args.n_layers,
#                        args.d_model, args.d_ff, args.n_heads, args.dropout)
#     BOS = english_tokenizer_load().bos_id()  # 2
#     EOS = english_tokenizer_load().eos_id()  # 3
#     src_tokens = [[BOS] + english_tokenizer_load().EncodeAsIds(sent) + [EOS]]
#     batch_input = torch.LongTensor(np.array(src_tokens)).to(args.device)
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

    parser = ArgumentParser()
    """ file path related """
    parser.add_argument("--data_dir", type=str, default='./data', \
                        help="Path to sentence piece model & vocab dir")
    parser.add_argument("--train_ch_data_path", type=str, default='./data/corpus/train.zh', \
                        help="training data file path")
    parser.add_argument("--train_en_data_path", type=str, default='./data/corpus/train.en', \
                        help="training data file path")
    parser.add_argument("--test_ch_data_path", type=str, default='./data/corpus/test.zh', \
                        help="test data file path")
    parser.add_argument("--test_en_data_path", type=str, default='./data/corpus/test.en', \
                        help="test data file path")
    parser.add_argument("--dev_ch_data_path", type=str, default='./data/corpus/valid.zh', \
                        help="dev data file path")
    parser.add_argument("--dev_en_data_path", type=str, default='./data/corpus/valid.en', \
                        help="dev data file path")
    parser.add_argument("--model_path", type=str, default='./output/model.pth', \
                        help="model save path")
    parser.add_argument("--model_path_best", type=str, default='./output/model_best.pth', \
                        help="best model save path")
    parser.add_argument("--temp_dir", type=str, default='./output/temp', \
                        help="temp data dir path")
    parser.add_argument("--log_path", type=str, default='./output/train.log', \
                        help="log save path")
    parser.add_argument("--output_path", type=str, default='./output/output.txt', \
                        help="test predict file path")
    """ train hypermeter related """
    parser.add_argument("--batch_size", type=int, default=14, help="Dataloader batch size")
    parser.add_argument("--epoch_num", type=int, default=40, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    # parser.add_argument("--seed", type=int, default=42, help='Seed for random init')
    parser.add_argument("--continue_training", type=bool, default=False, \
                        help="Whether to continue Training")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, \
                        help="Warmup proportion")
    parser.add_argument("--beam_size", type=int, default=3, \
                        help="beam size for decode")
    parser.add_argument("--max_len", type=int, default=60, \
                        help="max_len for decode")
    """ Transformer encoder-decoder design related """
    parser.add_argument("--d_model", type=int, default=512, help="")
    parser.add_argument("--n_heads", type=int, default=8, help="")
    parser.add_argument("--n_layers", type=int, default=6, help="")
    parser.add_argument("--d_k", type=int, default=64, help="")
    parser.add_argument("--d_v", type=int, default=64, help="")
    parser.add_argument("--d_ff", type=int, default=2048, help="")
    parser.add_argument("--dropout", type=float, default=0.1, help="")
    parser.add_argument("--padding_idx", type=int, default=0, help="")
    parser.add_argument("--bos_idx", type=int, default=2, help="")
    parser.add_argument("--eos_idx", type=int, default=3, help="")
    parser.add_argument("--src_vocab_size", type=int, default=32000, help="")
    parser.add_argument("--tgt_vocab_size", type=int, default=32000, help="")
    """ Distributed Training args """
    parser.add_argument("--n_gpu", type=int, default=8, help='Number of GPUs in one node')
    parser.add_argument("--n_node", type=int, default=2, help='Number of nodes in total')
    parser.add_argument("--node_rank", type=int, default=0, help='Node rank for this machine. 0 for master, and 1,2... for slaves')
    parser.add_argument("--MASTER_ADDR", type=str, default='10.104.91.31', help='Master Address')
    parser.add_argument("--MASTER_PORT", type=str, default='29501', help='Master port')

    args = parser.parse_args()
    print(args)

    os.environ['MASTER_ADDR'] = args.MASTER_ADDR
    os.environ['MASTER_PORT'] = args.MASTER_PORT

    mp.spawn(run, args=(args, ), nprocs=args.n_gpu)
    # translate_example()
