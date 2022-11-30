#from timeit import default_timer as timer
import transformer as tf

import torch
from torch import nn


def train_epoch(model, optimizer, encoder_input, valid_len, dev, PAD_IDX):
    model.train()

    src = encoder_input.to(dev)
    tgt = encoder_input.to(dev)

    tgt_input = torch.cat((torch.zeros(tgt[0].shape).unsqueeze(0), tgt[:-1, :]), dim=0)

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = tf.create_mask(src, tgt_input, dev, PAD_IDX)

    src_embedded, logits, embedding = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask,
                              src_padding_mask)

    optimizer.zero_grad()

    tgt_out = src_embedded
    criterion = nn.MSELoss()
    loss = criterion(logits, tgt_out)
    loss.backward()
    optimizer.step()

    return loss.item(), embedding


def evaluate(model, encoder_input, valid_len, dev, PAD_IDX):
    model.eval()

    src = encoder_input.to(dev)
    tgt = encoder_input.to(dev)

    tgt_input = torch.cat((torch.zeros(tgt[0].shape).unsqueeze(0), tgt[:-1, :]), dim=0)

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = tf.create_mask(src, tgt_input, dev, PAD_IDX)

    src_embedded, logits, embedding = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask,
                                            src_padding_mask)

    tgt_out = src_embedded
    criterion = nn.MSELoss()
    loss = criterion(logits, tgt_out)

    return loss.item()


def train(transformer, optimizer, encoder_inputs, valid_len, PAD_IDX, dev):
    encoder_inputs = torch.transpose(encoder_inputs, 0, 1)

    train_loss, embedding = train_epoch(transformer, optimizer, encoder_inputs, valid_len, dev, PAD_IDX)
    val_loss = evaluate(transformer, encoder_inputs, valid_len, dev, PAD_IDX)

    return embedding, train_loss, val_loss


def build_model(frame_dim, cell_num, dev):
    torch.manual_seed(0)

    SRC_VOCAB_SIZE = cell_num #10000
    TGT_VOCAB_SIZE = cell_num
    NUM_EMB_SIZE = 19  #原来是19
    EMB_SIZE = NUM_EMB_SIZE + frame_dim - 1 #48
    NHEAD = 6
    FFN_HID_DIM = 64
    # BATCH_SIZE = 128
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3

    transformer = tf.Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, NUM_EMB_SIZE, EMB_SIZE,
                                        NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    transformer = transformer.to(dev)

    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)

    return transformer, optimizer
