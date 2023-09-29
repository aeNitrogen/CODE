import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.Informer
import models.DLinear
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.c_out = configs.c_out
        self.revin = configs.revin
        self.wholeseq = configs.enc_len + configs.pred_len
        # encoder

        self.decoder = models.Informer.Model(configs)

        configs.__setattr__("enc_in", self.channels + configs.action_size)
        if self.wholeseq > 0:
            configs.__setattr__("seq_len", self.wholeseq)
        self.Encoder = models.DLinear.Model(configs)
    def get_sub(self):
        return self.Encoder

    def get_dec(self):
        return self
        # return self.decoder

    def forward(self, x, y, x_dec, y_dec):

        enc_out = self.Encoder(x)
        enc_out = enc_out[:, :, -self.c_out:]
        dec_out = self.forward_train(enc_out, y, x_dec, y_dec)

        return dec_out

    def forward_train(self, enc_out, y, x_dec, y_dec):
        # first input enc in, second dec in
        if self.revin == 0:
            dec_out = self.decoder(x_dec, y, enc_out, y_dec)
        else:
            dec_out = self.decoder(enc_out, y, x_dec, y_dec)
        return dec_out
