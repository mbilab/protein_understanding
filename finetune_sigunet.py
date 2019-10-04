from bert.train.train_sigunet import finetuneSigunet

import torch
import datetime
import json

pretrained_checkpoint = 'models/pretrain/SignalP_euk/checkpoint'
data_dir = None
dictionary_path = 'dic/dic.txt'
dataset_limit = None
epochs = 300
batch_size = 128
print_every = 1
save_every = 5
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

if __name__ == '__main__':

    for i in range(5):
        model_name = f'Sigunet_msk15_{i}'
        run_name = f"{model_name}:{layers_count}-hidden_size:{hidden_size}-heads_count:{heads_count}-fixed:{pretrain_fixed}-timestamp:{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        train_path = f'data/finetune_features/SignalP/euk_train_{i}.txt'
        val_path = f'data/finetune_features/SignalP/euk_valid_{i}.txt'
        checkpoint_dir = f'models/finetune/{run_name}'
        log_output = f'log/{run_name}.log'

        trainer = finetuneSigunet(pretrained_checkpoint,\
                data_dir, train_path, val_path, dictionary_path,\
                vocabulary_size, batch_size, max_len, epochs,\
                lr, clip_grads, device, layers_count, hidden_size, heads_count,\
                d_ff, dropout_prob, log_output, checkpoint_dir, print_every,\
                save_every, config, model_name, pretrain_fixed=pretrain_fixed)

        best_model_config = {
            'best_checkpoint_path': trainer.best_checkpoint_output_path,
            'layers_count': layers_count,
            'hidden_size': hidden_size,
            'heads_count': heads_count,
            'd_ff': d_ff,
            'dropout_prob': dropout_prob,
            'max_len': max_len,
            'vocabulary_size': vocabulary_size,
        }

        with open(f'models/finetune/{run_name}/best_model_config.json', 'w') as f:
                json.dump(best_model_config, f)

