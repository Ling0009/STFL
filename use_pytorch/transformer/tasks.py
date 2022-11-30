import copy
import math
import os
import matplotlib.pyplot as plt

import pandas as pd
import torch
from torch import nn
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import nni
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter

from num_embed import embedding
from utils import weight_init, gen_aligned_batch, gen_aligned_tensor, next_batch, cal_classify_metric, cal_regressive_metric, mean_absolute_percentage_error

import transformer as tf
# import models.reformer as rf
# import models.synthesizer as st
# import models.funnel as fn

exp_id = nni.get_experiment_id()
trial_id = nni.get_trial_id()
#encoder_cache = os.path.join('data', 'cache', 'dest_pre_encoder-{}-{}.model'.format(exp_id, trial_id))
#decoder_cache = os.path.join('data', 'cache', 'dest_pre_decoder-{}-{}.model'.format(exp_id, trial_id))
encoder_cache = "D:\周雪涵的课程文件 大三上\大创模型\实战演练\FedAvg-master\data\cache"+'dest_pre_encoder-{}-{}.model'.format(exp_id, trial_id)
decoder_cache = "D:\周雪涵的课程文件 大三上\大创模型\实战演练\FedAvg-master\data\cache"+'dest_pre_decoder-{}-{}.model'.format(exp_id, trial_id)
def _cal_batch_valid_len(input_batch, fill_value=0):
    valid_len = np.array([len(s) for s in input_batch])
    return gen_aligned_batch(input_batch, fill_value=fill_value), valid_len


def test_dest_pre(encoder_model, decoder_model, train_seq, val_seq, test_seq,
                  num_epoch, init_lr, device, batch_size, finetune,
                  early_stopping_round,frame_dim=0, max_n_steps=0, model_type=0):
    """
    Train and test a decoder to predict the destination of a trajectory given its encoded latent representation.

    :param encoder_model: an encoder instance to encode trajectories into latent representations.
    :param decoder_model: an decoder instance to predict the destination of trajectories.
    :param train_seq: a list containing all the training sequences.
    :param val_seq: a list containing all the validation sequences.
    :param test_seq: a list containing all the test sequences.
    :param num_epoch: number of training epochs.
    :param init_lr: initial learning rate.
    :param device: the name of device to put model parameters and inputs on.
    :param finetune: whether to finetune the encoder during the training of the decoder.
    :param early_stopping_round: number of continuous validation descending round before early stopping.
    :return: an series containing the best metric scores in this experiment.
    """
    encoder_model = encoder_model.to(device)
    decoder_model = decoder_model.to(device)
    optimizer = torch.optim.Adam(list(encoder_model.parameters()) + list(decoder_model.parameters()), lr=0.001)
    loss_func = nn.CrossEntropyLoss()

    early_stopping_round = 1000

    def _pre_batch(encoder_model, decoder_model, input_batch):
        """ Given prediction of one batch. """
        dest_label = [s[-1][0] for s in input_batch]
        input_batch, valid_len = gen_aligned_tensor(input_batch, fill_value=0)
        input_batch = torch.from_numpy(input_batch).float().to(device)  # (bs, max_valid_len, num_feat)
        valid_len = torch.from_numpy(valid_len).long().to(device)  # (bs)
        dest_label = torch.tensor(dest_label).long().to(device)  # (bs)

        if model_type == 1:
            seq_len = input_batch.size(1)
            src = torch.transpose(input_batch, 0, 1).to(device)
            tgt_input = input_batch[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = tf.create_mask(src, tgt_input, device, PAD_IDX=0)
            valid_len = valid_len - 1
            src_padding_mask = torch.arange(seq_len)[None, :].to(valid_len.device) >= valid_len[:, None]
            src = embedding(encoder_model.src_embed, src)
            seq_encoded = encoder_model.encode(src, src_mask=src_mask, src_key_padding_mask=src_padding_mask)

            if not decoder_model.is_transformer:
                output_size = src.size(2)
                seq_encoded = torch.transpose(seq_encoded, 0, 1)
                # _last_valid_index = (torch.arange(valid_len.size(0)).to(valid_len.device) * seq_len) + valid_len - 1  # (N)
                # seq_encoded = seq_encoded.reshape(-1, output_size)  # (N * L, output_size)
                # seq_encoded = seq_encoded[_last_valid_index.long()]  # (N, output_size)
                seq_encoded = seq_encoded.masked_fill_(src_padding_mask.unsqueeze(-1), 0)
                seq_encoded = seq_encoded.sum(1)  # (N, output_size)
                seq_encoded = seq_encoded / valid_len.unsqueeze(-1)  # (N, output_size)

        elif model_type == 2:
            src = torch.transpose(input_batch, 0, 1).to(device)
            _, hidden, _, _ = encoder_model.encoder(src, valid_len-1)  # csy modified 21/9/4 取hidden作为嵌入
            seq_encoded = hidden.squeeze(0)

        elif model_type == 3:
            src = torch.transpose(input_batch, 0, 1).to(device)
            seq_encoded = encoder_model.enc(src)
            seq_encoded = seq_encoded[-1]

        elif model_type == 4:
            src = torch.transpose(input_batch, 0, 1).to(device)
            src_mask = st.get_pad_mask(src, encoder_model.src_pad_idx)
            seq_encoded, *_ = encoder_model.encoder(src, src_mask)
            seq_encoded = seq_encoded[-1]

        elif model_type == 5:
            src = torch.transpose(input_batch, 0, 1).to(device)
            mask = fn.get_pad_mask(src, PAD_IDX=0)
            seq_encoded = encoder_model.encoder(src, input_mask=mask)
            seq_encoded = seq_encoded[-1][-1]

        elif model_type == 6:
            src = torch.transpose(input_batch, 0, 1).to(device)
            seq_encoded, _ = encoder_model.encoder(src)
            seq_encoded = seq_encoded[-1]

        elif model_type == 7:
            seq_encoded, _ = encoder_model(input_batch)
            seq_encoded = torch.transpose(torch.cat((seq_encoded[0], seq_encoded[1]), dim=2), 0, 1)
            seq_encoded = seq_encoded[-1]

        elif model_type == 8:
            src = torch.transpose(input_batch, 0, 1).to(device)
            seq_encoded, _, _ = encoder_model.encoder(src)
            seq_encoded = seq_encoded

        elif model_type == 0:
            _, seq_encoded = encoder_model(input_batch, valid_len-1)  # (N, embed_size)

        if not finetune:
            seq_encoded = seq_encoded.detach()  # Detach the output of encoder if not to finetune the encoder.
        dest_pre = decoder_model(seq_encoded)  # (batch_size, num_loc)

        return dest_pre, dest_label

    def _test_epoch(encoder_model, decoder_model, input_set):
        """ Test model on the whole input set. """
        encoder_model.eval()
        decoder_model.eval()
        dest_pres, dest_labels = [], []
        for batch in next_batch(input_set, batch_size):
            dest_pre, dest_label = _pre_batch(encoder_model, decoder_model, batch)
            dest_pres.append(dest_pre.detach().cpu().numpy())
            dest_labels.append(dest_label.detach().cpu().numpy())
        dest_pres = np.concatenate(dest_pres)  # (set_size, num_loc)
        dest_labels = np.concatenate(dest_labels)  # (set_size)
        return dest_pres, dest_pres.argmax(-1), dest_labels,

    max_metric = -0.1
    worse_round = 0
    results = []
    writer = SummaryWriter("./logs/des_pre_log")
    for epoch in range(num_epoch):
        losses = 0
        for batch in next_batch(shuffle(train_seq), batch_size):
            # Set the encoder and decoder to train mode. Their dropout layers will be activated.
            encoder_model.train()
            decoder_model.train()
            dest_pre, dest_label = _pre_batch(encoder_model, decoder_model, batch)
            optimizer.zero_grad()
            loss = loss_func(dest_pre, dest_label)
            loss.backward()
            optimizer.step()

            losses += loss.item()
        writer.add_scalars('loss', {'losses': losses}, epoch)
        if decoder_model.flag == 1:
            writer.add_histogram('dest_pre_linear1_weight', decoder_model.decoder[1].weight, epoch)
            writer.add_histogram('dest_pre_linear1_bias', decoder_model.decoder[1].bias, epoch)
            writer.add_histogram('dest_pre_linear2_weight', decoder_model.decoder[4].weight, epoch)
            writer.add_histogram('dest_pre_linear2_bias', decoder_model.decoder[4].bias, epoch)
        else:
            writer.add_histogram('dest_pre_linear_weight', decoder_model.decoder.weight, epoch)
            writer.add_histogram('dest_pre_linear_bias', decoder_model.decoder.bias, epoch)

        # After an epoch of training, test on the validation set to apply early stopping.
        dest_pres, dest_pres_index, dest_labels = _test_epoch(encoder_model, decoder_model, val_seq)
        metric = accuracy_score(dest_labels, dest_pres_index)
        s1=dest_labels
        s2=dest_pres_index
        x=[]
        y=[]
        colors=['FF9655','3EFF68','66F5FF','FDFF4F']
        # for i in range(0,s1.shape[0]):
        #     yi=s1[i]%50
        #     xi=(s1[i]-yi)/50
        #     plt.scatter(xi,yi,c=plt.cm.Set1(i%8))
        #     #plt.annotate(i, xy=(xi,yi),xytext=(xi+0.1,yi+0.1))
        # for i in range(0,s2.shape[0]):
        #     yi=s2[i]%50
        #     xi=(s2[i]-yi)/50
        #     plt.scatter(xi,yi,c=plt.cm.Set1(i%8))
            #plt.annotate(i, xy=(xi, yi), xytext=(xi + 0.1, yi + 0.1))
        # for i in range(0,s2.shape[0]):
        #     plt.annotate(i,xy=())
        #     plt.annotate(i, xy=(xi, yi), xytext=(xi + 0, 1, yi + 0.1))
        plt.show()

        results.append(metric)
        nni.report_intermediate_result(metric)
        if early_stopping_round > 0 and max_metric < metric:
            max_metric = metric
            worse_round = 0
            torch.save(encoder_model.state_dict(), encoder_cache)
            torch.save(decoder_model.state_dict(), decoder_cache)
        else:
            worse_round += 1

        if 0 < early_stopping_round <= worse_round:
            print('Early stopping @ epoch %d' % (epoch - worse_round), flush=True)
            break

    #plt.show()
    if early_stopping_round > 0:
        encoder_model.load_state_dict(torch.load(encoder_cache))
        decoder_model.load_state_dict(torch.load(decoder_cache))
    dest_pres, dest_pres_index, dest_labels = _test_epoch(encoder_model, decoder_model, test_seq)
    score_series = cal_classify_metric(dest_pres, dest_pres_index, dest_labels, [1, 5, 10, 20])
    print(score_series, flush=True)
    nni.report_final_result(score_series.loc['acc@1'])

    os.remove(encoder_cache)
    os.remove(decoder_cache)
    return score_series, results


def test_trajectory_similarity(encoder_model, test_seq, device, batch_size, distance_method,
                               model_type=0):
    """
    Test the quality of trajectory embeddings by utilizing them for trajectory similarity measurement.
    The similarity score is calculated by the Euclidean distance between two embedding vectors.

    :param encoder_model: an encoder instance to encode trajectories into latent representation.
    :param test_seq: a list containing all the testing sequences.
    :param device: the name of device to put model parameters and inputs on.
    :param distance_method: the name of method for distance measurement.
        Choose between 'euc' (Euclidean distance) or 'dot' (dot product)
    :return: an series containing the final metric scores in this experiment.
    """
    encoder_model = encoder_model.to(device)

    def _encode_batch(encoder_model, input_batch):
        """ Given encoded embeddings of one batch. """
        input_batch, valid_len = gen_aligned_tensor(input_batch, fill_value=0)
        input_batch = torch.from_numpy(input_batch).float().to(device)
        valid_len = torch.from_numpy(valid_len).long().to(device)

        if model_type == 1:
            src = input_batch
            tgt_input = input_batch[:, 1:]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = tf.create_mask(src, tgt_input, device, PAD_IDX=0)
            seq_encoded = encoder_model.encode(src, src_mask=src_mask, src_key_padding_mask=src_padding_mask)
            seq_encoded = torch.transpose(seq_encoded, 0, 1)
            seq_encoded = seq_encoded[-1]

        elif model_type == 2:
            src = torch.transpose(input_batch, 0, 1).to(device)
            _, _, _, seq_encoded = encoder_model.encoder(src)
            seq_encoded = seq_encoded[-1]

        elif model_type == 3:
            src = torch.transpose(input_batch, 0, 1).to(device)
            seq_encoded = encoder_model.enc(src)
            seq_encoded = seq_encoded[-1]

        elif model_type == 4:
            src = torch.transpose(input_batch, 0, 1).to(device)
            src_mask = st.get_pad_mask(src, encoder_model.src_pad_idx)
            seq_encoded, *_ = encoder_model.encoder(src, src_mask)
            seq_encoded = seq_encoded[-1]

        elif model_type == 5:
            src = torch.transpose(input_batch, 0, 1).to(device)
            mask = fn.get_pad_mask(src, PAD_IDX=0)
            seq_encoded = encoder_model.encoder(src, input_mask=mask)
            seq_encoded = seq_encoded[-1][-1]

        elif model_type == 6:
            src = torch.transpose(input_batch, 0, 1).to(device)
            seq_encoded, _ = encoder_model.encoder(src)
            seq_encoded = seq_encoded[-1]

        elif model_type == 7:
            seq_encoded, _ = encoder_model(input_batch)
            seq_encoded = torch.transpose(torch.cat((seq_encoded[0], seq_encoded[1]), dim=2), 0, 1)
            seq_encoded = seq_encoded[-1]

        elif model_type == 8:
            src = torch.transpose(input_batch, 0, 1).to(device)
            seq_encoded, _, _ = encoder_model.encoder(src)
            seq_encoded = seq_encoded

        elif model_type == 0:
            _, seq_encoded = encoder_model(input_batch, valid_len)  # (N, embed_size)
        return seq_encoded

    encoder_model.eval()
    even_sample, odd_sample = [s[::2] for s in test_seq], [s[1::2] for s in test_seq]

    _encoded_list = []
    for batch in next_batch(even_sample + odd_sample, batch_size):
        _encoded_list.append(_encode_batch(encoder_model, batch).detach())
    _encoded_list = torch.cat(_encoded_list, 0)
    even_encoded, odd_encoded = _encoded_list[:len(even_sample)], _encoded_list[len(even_sample):]  # (N, embed_size)

    distances = []
    for even_encoded_row in even_encoded:
        even_encoded_row = even_encoded_row.unsqueeze(0).repeat(odd_encoded.size(0), 1)  # (N, embed_size)
        if distance_method == 'euc':
            distance_row = (even_encoded_row - odd_encoded)  # (N, embed_size)
            distance_row = 1 / (distance_row * distance_row).sum(-1)  # (N)
        else:
            distance_row = (even_encoded_row * odd_encoded).sum(-1)   # (N)
        distance_row = distance_row / distance_row.sum()
        distances.append(distance_row)

    distances = torch.stack(distances, 0).cpu().numpy()  # (N, N)
    label = np.arange(odd_encoded.size(0)).astype(int)
    score_series = cal_classify_metric(distances, distances.argmax(-1), label, [1, 5, 10, 20])
    print(score_series)
    nni.report_final_result(score_series.loc['acc@1'])
    return score_series


def test_eta(encoder_model, decoder_model, train_seq, val_seq, test_seq, time_divisor,
             num_epoch, init_lr, device, batch_size, finetune,
             early_stopping_round, model_type=0):
    """
    Train and test a decoder to predict the arrival time of a trajectory given its encoded latent representation.

    :param encoder_model: an encoder instance to encode trajectories into latent representations.
    :param decoder_model: an decoder instance to estimate the arrival time of trajectories.
    :param train_seq: a list containing all the training sequences.
    :param val_seq: a list containing all the validation sequences.
    :param test_seq: a list containing all the test sequences.
    :param time_divisor: the divisor used to divide the time values.
    :param num_epoch: number of training epochs.
    :param init_lr: initial learning rate.
    :param device: the name of device to put model parameters and inputs on.
    :param finetune: whether to finetune the encoder during the training of the deocder.
    :param early_stopping_round: number of continuous validation descending round before early stopping.
    :return: an series containing the best metric scores in this experiment.
    """
    encoder_model = encoder_model.to(device)
    decoder_model = decoder_model.to(device)
    optimizer = torch.optim.Adam(list(encoder_model.parameters()) + list(decoder_model.parameters()), lr=init_lr)
    loss_func = nn.MSELoss()

    def _pre_batch(encoder_model, decoder_model, input_batch):
        time_label = [s[-1][1] - s[0][1] for s in input_batch]
        if not model_type == 0:
            input_batch = [s[:-1] for s in input_batch]
        input_batch, valid_len = gen_aligned_tensor(input_batch, fill_value=0)
        input_batch = torch.from_numpy(input_batch).float().to(device)  # (bs, max_valid_len, num_feat)
        valid_len = torch.from_numpy(valid_len).long().to(device)  # (bs)
        time_label = torch.tensor(time_label).float().to(device) / (time_divisor if model_type==0 else 1)  # (bs)

        if model_type == 1:
            src = input_batch
            tgt_input = input_batch[:, 1:]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = tf.create_mask(src, tgt_input, device, PAD_IDX=0)
            seq_encoded = encoder_model.encode(src, src_mask=src_mask, src_key_padding_mask=src_padding_mask)
            seq_encoded = torch.transpose(seq_encoded, 0, 1)
            seq_encoded = seq_encoded[-1]

        elif model_type == 2:
            src = torch.transpose(input_batch, 0, 1).to(device)
            _, _, _, seq_encoded = encoder_model.encoder(src)
            seq_encoded = seq_encoded[-1]

        elif model_type == 3:
            src = torch.transpose(input_batch, 0, 1).to(device)
            seq_encoded = encoder_model.enc(src)
            seq_encoded = seq_encoded[-1]

        elif model_type == 4:
            src = torch.transpose(input_batch, 0, 1).to(device)
            src_mask = st.get_pad_mask(src, encoder_model.src_pad_idx)
            seq_encoded, *_ = encoder_model.encoder(src, src_mask)
            seq_encoded = seq_encoded[-1]

        elif model_type == 5:
            src = torch.transpose(input_batch, 0, 1).to(device)
            mask = fn.get_pad_mask(src, PAD_IDX=0)
            seq_encoded = encoder_model.encoder(src, input_mask=mask)
            seq_encoded = seq_encoded[-1][-1]

        elif model_type == 6:
            src = torch.transpose(input_batch, 0, 1).to(device)
            seq_encoded, _ = encoder_model.encoder(src)
            seq_encoded = seq_encoded[-1]

        elif model_type == 7:
            seq_encoded, _ = encoder_model(input_batch)
            seq_encoded = torch.transpose(torch.cat((seq_encoded[0], seq_encoded[1]), dim=2), 0, 1)
            seq_encoded = seq_encoded[-1]

        elif model_type == 8:
            src = torch.transpose(input_batch, 0, 1).to(device)
            seq_encoded, _, _ = encoder_model.encoder(src)
            seq_encoded = seq_encoded

        elif model_type == 0:
            _, seq_encoded = encoder_model(input_batch, valid_len - 1)  # (N, embed_size)

        if not finetune:
            seq_encoded = seq_encoded.detach()
        time_pre = decoder_model(seq_encoded).squeeze(-1)

        return time_pre, time_label

    def _test_epoch(encoder_model, decoder_model, input_set):
        """ Test model on the whole input set. """
        encoder_model.eval()
        decoder_model.eval()
        time_pres, time_labels = [], []
        for batch in next_batch(input_set, batch_size):
            time_pre, time_label = _pre_batch(encoder_model, decoder_model, batch)
            time_pres.append(time_pre.detach().cpu().numpy())
            time_labels.append(time_label.detach().cpu().numpy())
        time_pres = np.concatenate(time_pres)
        time_labels = np.concatenate(time_labels)
        return time_pres, time_labels

    min_metric = 1e12

class CellClassifyDecoder(nn.Module):
    def __init__(self, input_size, output_size, max_len, is_transformer=False):
        """
        :param input_size: the vector dimension of input items.
        :param output_size: output size of the FC layer.
        """
        super().__init__()
        self.is_transformer = is_transformer
        self.max_len = max_len
        self.in_proj = nn.Linear(max_len, 1)
        self.decoder = nn.Sequential(nn.BatchNorm1d(input_size),
                                     nn.Linear(input_size, input_size * 4),
                                     nn.LeakyReLU(),
                                     nn.BatchNorm1d(input_size * 4),
                                     nn.Linear(input_size * 4, output_size),
                                     nn.Softmax(dim=-1))
        self.flag = 1
        self.apply(weight_init)

    def forward(self, x):
        """
        :param x: input state, shape (*, input_size)
        :return: decode result of the input, shape (*, output_size)
        """
        if self.is_transformer:
            #x=x.transpose(0,1)
            if x.shape[0] < self.max_len:
                x = torch.cat((torch.zeros(x.shape[1:]).unsqueeze(0).expand(
                    [self.max_len - x.shape[0], x.shape[1], x.shape[2]]), x),
                              dim=0)  # (L,N,frame_dim)
            else:
                print(F'max_len is too large:{x.shape[0]}')
            x = x.transpose(1, 2).transpose(0, 2)  # (N,frame_dim,L)
            x = self.in_proj(x).squeeze(2)
        hidden = self.decoder(x)
        return hidden
