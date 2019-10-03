from bert.train.train_sigunet import finetuneSigunet

import torch
import datetime

model_name='test_fixed'
pretrained_checkpoint = 'models/pretrain/SignalP_euk/checkpoint'
data_dir = None
train_path = 'data/finetune_features/SignalP/SignalP_euk_train.txt'
val_path = 'data/finetune_features/SignalP/SignalP_euk_val.txt'
dictionary_path = 'dic/dic.txt'
checkpoint_dir = f'models/finetune/{model_name}'
dataset_limit = None
epochs = 200
batch_size = 128
print_every = 1
save_every = 10
vocabulary_size = 30000
max_len = 1024
lr = 0.0001
clip_grads = 'store_true'
layers_count = 2
hidden_size = 128
heads_count = 2
d_ff = 128
dropout_prob = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = None
pretrain_fixed=True
run_name = f"Sigunet:{layers_count}-hidden_size:{hidden_size}-heads_count:{heads_count}-fixed:{pretrain_fixed}-timestamp:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
log_output = 'log/%s.log' % run_name

if __name__ == '__main__':
    finetuneSigunet(pretrained_checkpoint,\
            data_dir, train_path, val_path, dictionary_path,\
            vocabulary_size, batch_size, max_len, epochs,\
            lr, clip_grads, device, layers_count, hidden_size, heads_count,\
            d_ff, dropout_prob, log_output, checkpoint_dir, print_every,\
            save_every, config, model_name, pretrain_fixed=pretrain_fixed)
