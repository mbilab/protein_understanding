import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from bert.train import IGNORE_INDEX

def Seq2Seq_Metric(predictions, targets):
    return np.array(0.)

def Seq2Seq_preprocess(predictions, targets):
    _true = []
    _pred = []
    predictions = predictions.reshape(-1,2)
    targets = targets.reshape(-1)
    with open('./pred.txt','a')as f:
        f.write(f'{predictions}')
    for p, t in zip(predictions, targets):
        if IGNORE_INDEX == t:
            continue
        else:
            tmp = 1. if p[0] < p[1] else 0.
            _true.append(t)
            _pred.append(tmp)
    return _true, _pred

def Seq2Seq_MCC(predictions, targets):
    y_true, y_pred = Seq2Seq_preprocess(predictions, targets)
    return matthews_corrcoef(y_true, y_pred)

def Seq2Seq_ACC(predictions, targets):
    y_true, y_pred = Seq2Seq_preprocess(predictions, targets)
    return accuracy_score(y_true, y_pred)

def Seq2Seq_f1(predictions, targets):
    y_true, y_pred = Seq2Seq_preprocess(predictions, targets)
    return f1_score(y_true, y_pred)
