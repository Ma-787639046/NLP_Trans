import torch

d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1
padding_idx = 0
bos_idx = 2
eos_idx = 3
src_vocab_size = 32000
tgt_vocab_size = 32000
batch_size = 32
epoch_num = 40
early_stop = 5
lr = 3e-4

# greed decode的最大句子长度
max_len = 60
# beam size for bleu
beam_size = 3
# Warmup proportion
warmup_proportion = 0.1

data_dir = './data'
train_ch_data_path = './data/corpus/train.zh'
train_en_data_path = './data/corpus/train.en'
test_ch_data_path = './data/corpus/test.zh'
test_en_data_path = './data/corpus/test.en'
dev_ch_data_path = './data/corpus/valid.zh'
dev_en_data_path = './data/corpus/valid.en'
model_path = './output/model.pth'
log_path = './output/train.log'
output_path = './output/output.txt'

n_gpu = 8
n_node = 1
node_rank = 0
MASTER_ADDR='10.104.91.31'
MASTER_PORT='29501'