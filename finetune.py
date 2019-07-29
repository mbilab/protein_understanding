import torch
import datetime

from bert.preprocess.preprocess import build_dictionary
from bert.train.train import finetuneSPS

pretrained_checkpoint = 'checkpoint/model.pkl/epoch=1900-val_loss=2.71-val_metrics=0.184-0.473.pth'
data_dir = None
train_path = 'data/finetune/train.txt'
val_path = 'data/finetune/val.txt'
dictionary_path = 'dic/dic.txt'
checkpoint_dir = 'checkpoint/FinetuneModel'
dataset_limit = None
epochs = 100
batch_size = 16
print_every = 1
save_every = 10
vocabulary_size = 30000
max_len = 1024
lr = 0.001
clip_grads = 'store_true'
layers_count = 1
hidden_size = 128
heads_count = 2
d_ff = 128
dropout_prob = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = None
run_name = 'BERT-layers_count:%s-hidden_size:%s-heads_count:%s-timestamp:%s' % (\
        str(layers_count), str(hidden_size), str(heads_count),\
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
log_output = 'log/%s.log' % run_name


if __name__ == '__main__':
    finetuneSPS(pretrained_checkpoint,\
            data_dir, train_path, val_path, dictionary_path,\
            vocabulary_size, batch_size, max_len, epochs,\
            lr, clip_grads, device, layers_count, hidden_size, heads_count,\
            d_ff, dropout_prob, log_output, checkpoint_dir, print_every,\
            save_every, config, num_classes=8
            )
