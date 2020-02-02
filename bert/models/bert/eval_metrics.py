# ==================================================
#
# Copyright (c) 2018, Alibaba Inc.
# All Rights Reserved.
#
# ==================================================

# -*- coding: utf-8 -*-
"""
EvaluationMetrics class for evaluating input data.
Currently supported evaluation metrics:
    map,auc,mse,accuracy,ndcg@k,precision@k,recall@k
Created on Tues Apr 10 11:05:11 2018

"""
from __future__ import print_function

import random
import numpy as np
import math


def _to_list(x):
    if isinstance(x, list):
        return x
    return [x]

def corr(y_true, y_pred):
    from scipy import stats
    corr, pv = stats.pearsonr(y_pred, y_true)
    if np.isnan(corr):
        return 0.0
    else:
        return corr

def auc(y_true, y_pred):
    from sklearn import metrics
    try:
        auc = metrics.roc_auc_score(y_true.astype(int), y_pred)
    except Exception as e:
        print("compute auc exception", e)
        return 0.0
    return auc


def map(y_true, y_pred, rel_threshold=0):
    s = 0.
    y_true = _to_list(np.squeeze(y_true).tolist())
    y_pred = _to_list(np.squeeze(y_pred).tolist())
    c = list(zip(y_true, y_pred))
    random.shuffle(c)
    c = sorted(c, key=lambda x: x[1], reverse=True)
    ipos = 0
    for j, (g, p) in enumerate(c):
        if g > rel_threshold:
            ipos += 1.
            s += ipos / (j + 1.)
    if ipos == 0:
        s = 0.
    else:
        s /= ipos
    return s


def ndcg(k=10):
    def top_k(y_true, y_pred, rel_threshold=0.):
        if k <= 0.:
            return 0.
        s = 0.
        y_true = _to_list(np.squeeze(y_true).tolist())
        y_pred = _to_list(np.squeeze(y_pred).tolist())
        c = zip(y_true, y_pred)
        random.shuffle(c)
        c_g = sorted(c, key=lambda x: x[0], reverse=True)
        c_p = sorted(c, key=lambda x: x[1], reverse=True)
        idcg = 0.
        ndcg = 0.
        for i, (g, p) in enumerate(c_g):
            if i >= k:
                break
            if g > rel_threshold:
                idcg += (math.pow(2., g) - 1.) / math.log(2. + i)
        for i, (g, p) in enumerate(c_p):
            if i >= k:
                break
            if g > rel_threshold:
                ndcg += (math.pow(2., g) - 1.) / math.log(2. + i)
        if idcg == 0.:
            return 0.
        else:
            return ndcg / idcg
    
    return top_k


def precision(k=10):
    def top_k(y_true, y_pred, rel_threshold=0.5):
        if k <= 0:
            return 0.
        s = 0.
        y_true = _to_list(np.squeeze(y_true).tolist())
        y_pred = _to_list(np.squeeze(y_pred).tolist())
        c = zip(y_true, y_pred)
        random.shuffle(c)
        c = sorted(c, key=lambda x: x[1], reverse=True)
        ipos = 0
        prec = 0.
        for i, (g, p) in enumerate(c):
            if i >= k:
                break
            if g > rel_threshold:
                prec += 1.0
        prec /= k
        return prec
    
    return top_k


# compute recall@k
# the input is all documents under a single query
def recall(k=10):
    def top_k(y_true, y_pred, rel_threshold=0.):
        if k <= 0:
            return 0.
        s = 0.
        y_true = _to_list(np.squeeze(y_true).tolist())  # y_true: the ground truth scores for documents under a query
        y_pred = _to_list(np.squeeze(y_pred).tolist())  # y_pred: the predicted scores for documents under a query
        pos_count = sum(i > rel_threshold for i in y_true)  # total number of positive documents under this query
        c = zip(y_true, y_pred)
        random.shuffle(c)
        c = sorted(c, key=lambda x: x[1], reverse=True)
        ipos = 0
        recall = 0.
        for i, (g, p) in enumerate(c):
            if i >= k:
                break
            if g > rel_threshold:
                recall += 1
        
        if pos_count < 1:
            return 0.
        else:
            recall /= pos_count
            return recall
    
    return top_k


def precision_all(y_true, y_pred):
    k = 0.
    prec = 0.
    y_true = _to_list(np.squeeze(y_true).tolist())
    y_pred = _to_list(np.squeeze(y_pred).tolist())
    for (g, p) in zip(y_true, y_pred):
        if p > 0.0 and g > 0.5:
            prec += 1
        if p > 0.0:
            k += 1.0
    if k > 0.0:
        return prec / k
    else:
        return prec


def recall_all(y_true, y_pred):
    k = 0.
    rec = 0.
    y_true = _to_list(np.squeeze(y_true).tolist())
    y_pred = _to_list(np.squeeze(y_pred).tolist())
    for (g, p) in zip(y_true, y_pred):
        if p > 0.0 and g > 0.5:
            rec += 1
        if g > 0.5:
            k += 1.0
    if k > 0.0:
        return rec / k
    else:
        return rec


def mse(y_true, y_pred, rel_threshold=0.):
    return np.mean(np.square(y_pred - y_true), axis=-1)


def accuracy(y_true, y_pred):
    # y_true = _to_list(np.squeeze(np.int(y_true)).tolist())
    # y_pred = _to_list(np.squeeze(np.int(y_pred)).tolist())
    y_true_idx = np.argmax(y_true)
    y_pred_idx = np.argmax(y_pred)
    assert y_true_idx.shape == y_pred_idx.shape
    return 1.0 * np.sum(y_true_idx == y_pred_idx) / len(y_true)


def acc_multi_class(k=1):
    def accuracy_multi_class(y_true, y_pred):
        assert np.ndim(y_pred) == 2, "the dimension of predict results must be equal to 2"
        assert np.shape(y_pred)[1] >= k, "last dimension of predict results should not be less then param k"
        assert len(y_true) == len(y_pred), "label array and predict array should have the same length"
        acc_count = 0
        for i in range(len(y_true)):
            y_truncate = sorted(y_pred[i])[:k]
            if y_true[i] in y_truncate:
                acc_count += 1
        return 1.0 * acc_count / len(y_true)
    
    return accuracy_multi_class


class EvaluationMetrics(object):
    """
        EvaluationMetrics class for evaluating input data.
        Currently supported evaluation metrics:
            map,auc,mse,accuracy,ndcg@k,precision@k,recall@k
    """
    
    def __init__(self, eval_conf=['map', 'auc']):
        from collections import OrderedDict
        
        # define supported metrics
        eval_metric_dict = dict(
            map=map,
            auc=auc,
            mse=mse,
            accuracy=accuracy,
            precision_all=precision_all,
            recall_all=recall_all,
            corr=corr
        )
        
        for k in range(50):
            eval_metric_dict['ndcg@%d' % k] = ndcg(k)
            eval_metric_dict['precision@%d' % k] = precision(k)
            eval_metric_dict['recall@%d' % k] = recall(k)
            eval_metric_dict['acc_multi@%d' % k] = acc_multi_class(k)
        
        # build evaluation dict
        self.metrics = OrderedDict()
        for mobj in eval_conf:
            self.metrics[mobj] = eval_metric_dict[mobj]
    
    def eval_all(self, y_true, y_pred, list_counts):
        """
            Evaluation on a list of results separated by list_counts,
              e.g.: res['acc'][k] is defined as:
                eval_accuracy(y_true[list_counts[k]:list_counts[k+1]], y_pred[list_counts[k]:list_counts[k+1]])
                and, final_acc = average(res['acc]).
            Args:
                y_true: labels
                y_pred: prediction results
                list_counts: a list of indexes
        """
        
        res = dict([[k, 0.] for k in self.metrics.keys()])
        for k, eval_func in self.metrics.items():
            for lc_idx in range(len(list_counts) - 1):
                pre = list_counts[lc_idx]
                suf = list_counts[lc_idx + 1]
                # print('y_true: {}'.format(y_true))
                # print('y_pred: {}'.format(y_pred))
                res[k] += eval_func(y_true=np.array(y_true[pre:suf]),
                                    y_pred=np.array(y_pred[pre:suf]))
        
        num_valid = len(list_counts) - 1
        
        for k in res.keys():
            res[k] = res[k] / num_valid
        
        return res
    
    def eval_multi_class(self, y_true, y_pred):
        res = {}
        for k, eval_func in self.metrics.items():
            res[k] = eval_func(y_true=y_true, y_pred=y_pred)
        
        return res
