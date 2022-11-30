import math
from itertools import zip_longest

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error


def gen_index_map(series, offset=0):
    index_map = {origin: index + offset
                 for index, origin in enumerate(series.drop_duplicates())}
    return index_map


def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_index = (y_true > 0)
    y_true = y_true[non_zero_index]
    y_pred = y_pred[non_zero_index]

    mape = np.abs((y_true - y_pred) / y_true)
    mape[np.isinf(mape)] = 0
    return np.mean(mape) * 100


def top_n_accuracy(truths, preds, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]
    successes = 0
    for i, truth in enumerate(truths):
        if truth in best_n[i, :]:
            successes += 1
    return float(successes) / truths.shape[0]


def cal_classify_metric(pre_indices, pres, labels, top_n_list):
    """
    Calculate comprehensive classify metrics.

    :param pre_indices: the indices of prediction, or the indices of maximum values, shape (N)
    :param pres: the predicted classify distribution, shape (N, num_class)
    :param labels: the indices of labels, shape (N)
    :param top_n_list: a list containing all Ns for top-n accuracy calculation.
    :return: a pandas series containing all metrics.
    """
    precision, recall, f1 = precision_score(labels, pres, average='macro'), \
                            recall_score(labels, pres, average='macro'), \
                            f1_score(labels, pres, average='macro')
    if pre_indices is not None:
        top_n_acc = [top_n_accuracy(labels, pre_indices, n) for n in top_n_list]
    else:
        top_n_acc = [accuracy_score(labels, pres)] + [-1.0 for _ in range(len(top_n_list)-1)]
    score_series = pd.Series([precision, recall, f1] + top_n_acc,
                             index=['macro-pre', 'macro-rec', 'macro-f1'] + ['acc@{}'.format(n) for n in top_n_list])
    return score_series


def cal_regressive_metric(pres, labels):
    """
    Calculate comprehensive classify metrics.
    """
    mae, mape, rmse = mean_absolute_error(labels, pres), \
                      mean_absolute_percentage_error(labels, pres), \
                      math.sqrt(mean_squared_error(labels, pres))
    score_series = pd.Series([mae, mape, rmse], index=['mae', 'mape', 'rmse'])
    return score_series


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.Embedding):
        embed_size = m.weight.size(-1)
        if embed_size > 0:
            init_range = 0.5/m.weight.size(-1)
            init.uniform_(m.weight.data, -init_range, init_range)


def gen_aligned_batch(sequences, fill_value):
    """
    Generate an aligned batch from sequences with uneven length.

    :param sequences: a list containing the original sequences.
    :param fill_value: the value for filling the sequences.
    :return: an numpy array containing the aligned batch.
    """
    return np.transpose(np.array(list(zip_longest(*sequences, fillvalue=fill_value))))


def gen_aligned_tensor(sequences, fill_value, max_len=None):
    """
    Generate an aligned tensor given a list of tuples with uneven length.

    :param sequences: a list containing the original sequences.
    :param fill_value: the value for filling the sequences.
    :param max_len: the maximal length of the given sequence.
    :return:
    """
    aligned_sequences = []
    valid_lens = []
    if not max_len:
        if (len(sequences)>1):
            max_len = max(*[len(s) for s in sequences])
        else:
            max_len = len(sequences[0])
    tuple_len = len(sequences[0][0])
    empty_tuple = tuple([fill_value] * tuple_len)
    for sequence in sequences:
        valid_len = len(sequence)
        aligned_sequences.append(sequence + [empty_tuple] * (max_len - valid_len))
        valid_lens.append(valid_len)
    return np.array(aligned_sequences), np.array(valid_lens)


def fetch_valid_seq_data(input_seq, valid_len):
    """
    Fetch the valid part of an set of input sequence.

    :param input_seq: the input sequence with variable valid lengths, shape (N, max_len, E)
    :param valid_len: the valid lengths of every sequence in the input batch, shape (N)
    :return: the valid part of the input sequence, shape (total_num_valid, E)
    """
    packed_input_seq = pack_padded_sequence(input_seq, valid_len.to('cpu'), batch_first=True, enforce_sorted=False)
    return packed_input_seq.data


def gen_casual_mask(seq_len, include_self=True):
    """
    Generate a casual mask which prevents i-th output element from
    depending on any input elements from "the future".
    Note that for PyTorch Transformer model, sequence mask should be
    filled with -inf for the masked positions, and 0.0 else.

    :param seq_len: length of sequence.
    :return: a casual mask, shape (seq_len, seq_len)
    """
    if include_self:
        mask = 1 - torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1)
    else:
        mask = 1 - torch.tril(torch.ones(seq_len, seq_len)).transpose(0, 1)
    return mask.bool()


def lng2meter(lng):
    semimajoraxis = 6378137.0
    east = lng * 0.017453292519943295
    return semimajoraxis * east


def lat2meter(lat):
    north = lat * 0.017453292519943295
    t = np.sin(north)
    return 3189068.5 * np.log((1 + t) / (1 - t))


def grid_discretization(col_array, grid_length, noisy_meter=0):
    """
    Transform coordinates into discrete

    :param col_array: an numpy array with shape (N, 2), each row is a coordinate point (lng, lat).
    :param grid_length: the length of a grid's edge (meter).
    :param noisy_meter: length of noise added to the dataset. Will apply a noise_meter * N(0, 1) noise on the original coordinates.
    :return: an 1-d numpy array, containing corresponding grid index of given coordinates.
    """
    lng = lng2meter(col_array[:, 0])
    lat = lat2meter(col_array[:, 1])
    if noisy_meter > 0:
        lng += np.random.randn(lng.shape[0]) * noisy_meter
        lat += np.random.randn(lat.shape[0]) * noisy_meter
    lng_index = np.floor((lng - lng.min()) / grid_length)
    lat_index = np.floor((lat - lat.min()) / grid_length)
    num_lng_index = lng_index.max() + 1
    return lng_index * num_lng_index + lat_index
