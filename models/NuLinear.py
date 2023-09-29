import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import models.DLinear
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding,DataEmbedding_wo_pos,DataEmbedding_wo_temp,DataEmbedding_wo_pos_temp


class Model(nn.Module):

    def __init__(self, configs):
        print("Trans")
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.c_out = configs.c_out
        self.revin = configs.revin
        dec_in = 4
        # Embedding
        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            self.dec_embedding = DataEmbedding(dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            self.dec_embedding = DataEmbedding(dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(dec_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                       configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(dec_in, configs.d_model, configs.embed, configs.freq,
                                                       configs.dropout)
        elif configs.embed_type == 4:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                           configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(dec_in, configs.d_model, configs.embed, configs.freq,
                                                           configs.dropout)
        # encoder
        self.Encoder = models.DLinear.Model(configs)

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def get_sub(self):
        assert False, "deprecated"
        return self.Encoder

    def get_dec(self):
        assert False, "deprecated"
        return self.decoder

    def forward(self, x, y, x_dec, y_dec):

        enc_out = self.Encoder(x)
        enc_out = self.enc_embedding(enc_out, y)
        x_dec = self.dec_embedding(x_dec, y_dec)
        dec_out = self.decoder(x_dec, enc_out)

        return dec_out

    def forward_train(self, enc_out, y, x_dec, y_dec):

        enc_out = self.enc_embedding(enc_out, y)
        x_dec = self.dec_embedding(x_dec, y_dec)
        if self.revin == 1:
            dec_out = self.decoder(x_dec, enc_out)
        else:
            dec_out = self.decoder(enc_out, x_dec)
        return dec_out
