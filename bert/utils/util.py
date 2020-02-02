# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import division

import numpy as np
import tensorflow as tf
from tensorflow import logging
from train_flags import FLAGS
from models.bert import modeling


def evaluate(results, metrics, interval=None):
    if np.array(results[0]).shape[0] != np.array(results[1]).shape[0]:
        print('  [Evaluate %d instances]: ' % len(results[0]))
        print('\ty_pred shape: {}'.format(np.array(results[0]).shape))
        print('\ty_true shape: {}'.format(np.array(results[1]).shape))
        raise Exception

    # evaluation metrics
    if interval is not None:
        list_counts = np.linspace(0, len(results[0]), interval, dtype=int)
    else:
        list_counts = [0, len(results[0])]

    y_pred = np.array(results[0])
    y_true = np.array(results[1])
    res = metrics.eval_all(y_true=y_true, y_pred=y_pred,
                           list_counts=list_counts)

    # with open(method_conf['saver_spec']['directory'] + '/../predictions.txt', 'w') as fw:
    #     for tmp_pred, tmp_true in zip(y_pred, y_true):
    #         fw.write('%f %f\n' % (tmp_pred, tmp_true))

    # compute acc
    y_pred2 = (y_pred > 0.5).astype(np.int)
    acc = np.sum(y_pred2 == y_true.astype(np.int)) / len(y_pred2)
    res['accuracy'] = acc

    print('  Evaluated instances shape: {}'.format(np.array(results[0]).shape),
          '  '.join(['%s=%6f' % (k, v) for k, v in res.items()]))
    print('  y_pred, max {:.4} min {:.4} mean {:.4}'.format(float(np.max(y_pred)),
                                                            float(np.min(y_pred)),
                                                            float(np.mean(y_pred))))
    print('  y_true, max {:.4} min {:.4} mean {:.4}'.format(float(np.max(y_true)),
                                                            float(np.min(y_true)),
                                                            float(np.mean(y_true))))
    return res


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    assert len(np.array(x).shape) == 2
    e_x = np.exp(x - np.max(x))
    norm_e_x = e_x.T / e_x.T.sum(axis=0)
    return norm_e_x.T


def get_top_class(predictions):
    """
    input: predictions [None, N]
    :return: top class idx [None]
    """
    assert len(np.array(predictions).shape) == 2
    return np.argmax(predictions, axis=1)


def merge_gradients(tower_grad):
    grads_and_vars = tower_grad.pop()
    grad_len = float(len(tower_grad))
    # if len tower_grad > 0 means that more than
    # one gradients need to be averaged.
    if grad_len > 0:
        gs = []
        vs = []
        for i, (g, v) in enumerate(grads_and_vars):
            gs.append(g)
            vs.append(v)

        for grad in tower_grad:
            for i, (g, v) in enumerate(grad):
                assert v == vs[i]
                if isinstance(g, tf.Tensor):
                    gs[i] += g
                elif isinstance(g, tf.IndexedSlices):
                    sum_values = tf.accumulate_n([gs[i].values, g.values])
                    sum_indices = tf.accumulate_n([gs[i].indices, g.indices])
                    gs[i] = tf.IndexedSlices(sum_values, sum_indices, g.dense_shape)

        for i in range(len(gs)):
            if isinstance(gs[i], tf.Tensor):
                gs[i] /= grad_len + 1.0
            elif isinstance(gs[i], tf.IndexedSlices):
                gs[i] = tf.IndexedSlices(sum_values / grad_len + 1.0, sum_indices, gs[i].dense_shape)
        grads_and_vars = zip(gs, vs)
    return grads_and_vars


def load_checkpoint():
    tvars = tf.trainable_variables()
    import os
    init_checkpoint = os.path.join(FLAGS.model_dir, FLAGS.ckpt_file_name)
    print('start loading checkpoint with path: ', init_checkpoint)
    (assignment_map, initialized_variable_names) = \
        modeling.get_assigment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        else:
            # un_init_variables.add(var)
            pass
        logging.info("  name = %s, shape = %s, device = %s%s", var.name, var.shape, var.device, init_string)

