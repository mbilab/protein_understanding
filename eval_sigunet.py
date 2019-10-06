from bert.preprocess.dictionary import IndexDictionary
from bert.train.model.bert import build_model
from bert.train.datasets.NoOneHot import Seq2SeqDataset, SPDS17Dataset
from bert.train.utils.stateload import stateLoading
from bert.train.utils.convert import convert_to_tensor, convert_to_array
from bert.train.optimizers import NoamOptimizer
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from bert.train.sigunet.sigunet import sigunet
from bert.train.Seq2Seq.utils import Seq2Seq_Metric

import torch
from torch.nn import DataParallel, Softmax
from torch.utils.data import DataLoader

import json
import numpy as np
from os.path import join
import random
from sklearn.metrics import matthews_corrcoef
from sys import argv
from tqdm import tqdm

data_dir = None
dictionary_path = 'dic/dic.txt'
batch_size = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fixed_length = 96

def predict(model, dataloader):
    y_pred = []
    y_true = []

    for inputs, targets, batch_count in dataloader:
        inputs = convert_to_tensor(inputs, device)
        # targets = convert_to_tensor(targets, device)

        output, _ = model(inputs, 0, is_prediction=True)
        output = Softmax(dim=2)(output)
        output = convert_to_array(output)

        y_pred.append(output[:, :, 2])
        # y_true.append(target_classify(targets[0]))
        y_true.append(targets[0])

    return np.concatenate(y_pred, axis=0), np.array(y_true)

def build_dataloader(data_path, is_SPDS17=False):
    data_path = data_path if data_dir is None else join(data_dir, data_path)

    if is_SPDS17:
        dataset = SPDS17Dataset(data_path=data_path, dictionary=dictionary, fixed_length=fixed_length)
    else:
        dataset = Seq2SeqDataset(data_path=data_path, dictionary=dictionary, fixed_length=fixed_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_function)

    return dataloader

def target_classify(target):
    if type(target) == int:
        return 1 if target == 2 else -1
    elif len(target) == 1:
        return 1 if target == 2 else -1
    elif 2 in target:
        return 1
    else:
        return -1

def peptide_classify(peptide, n, thr):
    """Decide sequence is signal peptide or not.

    Return true if continuous `n` positions have probability of signal peptide is upper than `thr`.

    Arguments:
      sequence_proba (numpy array): A sequence probability of a protein.
      n (int): The size of continuous range.
      thr (float): The threshold to decide a position is signal peptide residue or not.

    Returns:
      is_signal_peptide (bool): `True` for signal peptide, `False` for other.
    """

    # Calculate accumlate sum array
    accumlate = []
    accumlate.append(1 if peptide[0] >= thr else 0)
    for i in range(1, len(peptide)):
        accumlate.append(accumlate[-1] + (1 if peptide[i] >= thr else 0))

    # append `0` for `accumlate[-1]`
    accumlate.append(0)
    # print(accumlate)
    for i in range(0, len(peptide) - n + 1):
        if accumlate[i + n - 1] - accumlate[i - 1] == n:
            return 1

    return -1

def find_n_thr(y_pred, y_ture):
    best_config={'n': 0, 'thr': 0, 'mcc': 0}
    for n in range(4,5):
        for thr100 in range(50, 90, 2):
            thr = thr100 / 100
            y_pred_ = [peptide_classify(y, n, thr) for y in y_pred]
            mcc = matthews_corrcoef(y_ture, y_pred_)
            if mcc > best_config['mcc']:
                best_config['n'] = n
                best_config['thr'] = thr
                best_config['mcc'] = mcc
                print(f"n, thr mcc = {n}, {thr}, {mcc}")

    return best_config['n'], best_config['thr'], best_config['mcc']

def load_model(conf, vocabulary_size):
    pretrained_model = build_model(
        conf['layers_count'],
        conf['hidden_size'],
        conf['heads_count'],
        conf['d_ff'],
        conf['dropout_prob'],
        conf['max_len'],
        vocabulary_size,
        forward_encoded=True)

    model = sigunet(pretrained_model, m=28, n=4, kernel_size=7, pool_size=2, threshold=0.1, device=device)

    state_dict = torch.load(conf['best_checkpoint_path'], map_location=device)
    model.load_state_dict(state_dict['state_dict'])

    return model.to(device)

if '__main__' == __name__:
    dictionary = IndexDictionary.load(dictionary_path=dictionary_path,
                                      vocabulary_size=30000)

    vocabulary_size = len(dictionary)


    y_pred = []
    y_true = []
    for i in range(5):
        model_dir = f'models/finetune/Sigunet_msk15_1_{i}:2-hidden_size:128-heads_count:2-fixed:True'
        valid_path = f'data/finetune_features/SignalP/euk_valid_{i}.txt'
        with open(f'{model_dir}/best_model_config.json') as json_data_file:
            conf = json.load(json_data_file)

        dataloader = build_dataloader(valid_path)
        model = load_model(conf, vocabulary_size)

        y_prob, y_label = predict(model, dataloader)
        y_label = np.array([target_classify(label) for label in y_label])

        y_pred.append(y_prob)
        y_true.append(y_label)

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)

    n, thr, mcc = find_n_thr(y_pred, y_true)
    print(f"n, thr mcc = {n}, {thr}, {mcc}")

    test_path = f'data/finetune_features/SignalP/euk_test.txt'
    test_dataloader = build_dataloader(test_path, is_SPDS17=True)

    y_pred = []
    for i in range(5):
        model_dir = f'models/finetune/Sigunet_msk15_1_{i}:2-hidden_size:128-heads_count:2-fixed:True'
        with open(f'{model_dir}/best_model_config.json') as json_data_file:
            conf = json.load(json_data_file)

        model = load_model(conf, vocabulary_size)

        y_prob, y_label = predict(model, test_dataloader)

        y_pred.append(y_prob)
        if 0 == i:
            y_true = y_label

    y_pred = np.mean(y_pred, axis=0)
    y_pred = [peptide_classify(y, n, thr) for y in y_pred]
    y_pred = np.array(y_pred)

    y_true = [target_classify(int(y)) for y in y_true]
    y_true = np.array(y_true)


    mcc = matthews_corrcoef(y_true, y_pred)
    print(mcc)
