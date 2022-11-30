import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer

import math

from num_embed import embedding

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 num_emb_size: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.emb_size = emb_size
        self.src_vocab_size=src_vocab_size
        self.num_emb_size=num_emb_size
        self.transformer = Transformer(d_model=emb_size, #60
                                       nhead=nhead, #6
                                       num_encoder_layers=num_encoder_layers, #3
                                       num_decoder_layers=num_decoder_layers, #3
                                       dim_feedforward=dim_feedforward, #512
                                       dropout=dropout) #0.1
        self.src_embed = nn.Embedding(src_vocab_size, num_emb_size)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, num_emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src = embedding(self.src_embed, src)
        tgt = embedding(self.tgt_embed, tgt)
        # lstm = torch.nn.LSTM(input_size=self.emb_size, hidden_size=48, num_layers=1,
        #                      bias=True, batch_first=False, dropout=0, bidirectional=False)
        # lstm_input = src
        # output, hn = lstm(lstm_input)
        #memory = self.encode(output, src_mask=src_mask, src_key_padding_mask=src_padding_mask)
        memory = self.encode(src, src_mask=src_mask, src_key_padding_mask=src_padding_mask)
        #print(memory.size)
        #加一个lstm层

        # lstm = torch.nn.LSTM(input_size=self.emb_size, hidden_size=48, num_layers=1,
        #                            bias=True, batch_first=False, dropout=0, bidirectional=False)
        # lstm_input = memory
        # output, hn = lstm(lstm_input)
        #为什么这里又decode了？
        # outs = self.decode(tgt, output, tgt_mask=tgt_mask,
        #                    tgt_key_padding_mask=tgt_padding_mask,
        #                    memory_key_padding_mask=memory_key_padding_mask)
        outs = self.decode(tgt, memory, tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
        # outs = self.generator(outs)
        return src, outs, memory

    def encode(self, src: Tensor, src_mask: Tensor, src_key_padding_mask: Tensor):
        # return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)
        return self.transformer.encoder(self.positional_encoding(src), src_mask, src_key_padding_mask)
    #这里，是如何进入encoder Layers的？

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor,
               tgt_key_padding_mask: Tensor,
               memory_key_padding_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
            tgt), memory, tgt_mask, None, tgt_key_padding_mask, memory_key_padding_mask)


# 重写，为了用device
def generate_square_subsequent_mask(sz, dev):
    mask = (torch.triu(torch.ones((sz, sz), device=dev)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt, dev, PAD_IDX):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, dev)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=dev).type(torch.bool)

    temp_src = torch.matmul(torch.abs(src), torch.ones((src.shape[-1], 1), device=dev)).squeeze(-1)
    temp_tgt = torch.matmul(torch.abs(tgt), torch.ones((tgt.shape[-1], 1), device=dev)).squeeze(-1)

    src_padding_mask = (temp_src == PAD_IDX * src.shape[-1]).transpose(0, 1)
    a = torch.zeros(tgt.shape[1]).type(torch.bool).unsqueeze(0)
    tgt_padding_mask = torch.cat((a, (temp_tgt == PAD_IDX * tgt.shape[-1])[1:, :]), dim=0).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask