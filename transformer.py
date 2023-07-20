import torch
import torch.nn as nn
import torch.optim as optim
import posEnc
import numpy as np
import time
from torch.nn import functional as F
import utils_ba

torch.manual_seed(8055)
native_batch_size = 750    # number of samples/batch size
native_train_batch = 500
batch_size = 30
block_size = 10     # maximum context length for predictions
L = 900  # length of each sample
T = 13  # input var count
Out_num = 9  # output var count
dType = torch.float32


class Transformer(nn.Module):

    def __init__(self, n_hidden_enc=256, n_hidden_dec=256, n_heads=4, dropout=0.1, forecasting_len=96, d_model=1024,
                 encoder_len=96, dec_overlap=5):
        super().__init__()
        self.dec_overlap = dec_overlap
        self.forecasting_len = forecasting_len
        self.train = True
        self.encoder_len = encoder_len
        self.encoder_input_layer = nn.Linear(T * encoder_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=n_hidden_enc, dropout=dropout,
                                               batch_first=True)

        self.decoder_input_layer = nn.Linear(T * (forecasting_len + dec_overlap), d_model)
        self.encoder = nn.TransformerEncoder(enc_layer, 6)

        dec_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=n_hidden_dec, dropout=dropout,
                                               batch_first=True)

        self.decoder = nn.TransformerDecoder(dec_layer, 6)

        self.lin_end = nn.Linear(d_model, (forecasting_len + dec_overlap) * Out_num)

    def train_mode(self):
        self.train = True
        self.encoder.train()
        self.decoder.train()
        self.encoder_input_layer.train()
        self.decoder_input_layer.train()
        self.lin_end.train()

    def eval_mode(self):
        self.train = False
        self.encoder.eval()
        self.decoder.eval()
        self.encoder_input_layer.eval()
        self.decoder_input_layer.eval()
        self.lin_end.eval()

    def forward(self, x: torch.tensor, targets: torch.tensor):
        # x is the encoder input, containing observations and actions
        # targets is the vector containing the actions for the target sequence
        temp = torch.zeros(x.size(0), self.forecasting_len, x.size(2))
        temp[:, :, :T - Out_num] = targets
        temp = torch.cat((x, temp), dim=1)
        temp = posEnc.fwd(temp, dr=self.train)
        x = temp[:, :x.size(1), :]
        dec_input = temp[:, -(self.forecasting_len + self.dec_overlap):, :]
        # encode
        x = x.flatten(1, 2)
        dec_input = dec_input.flatten(1, 2)
        # print(x.size())
        enc_in = self.encoder_input_layer.forward(x)
        enc_out = self.encoder.forward(enc_in)

        # create decoded tensor

        # decoded = torch.full_like((x.size(0), self.forecasting_len, x.size(2)), float("-inf"))

        # decode
        dec_input = self.decoder_input_layer.forward(dec_input)
        decoded = self.decoder.forward(dec_input, enc_out)
        dec_out = self.lin_end.forward(decoded)
        # print(dec_out.size())
        # print(dec_out.size(1) / T)
        dec_out = dec_out.unflatten(1, (dec_out.size(1) // Out_num, Out_num))
        # decoder finished
        # positional decoding
        dec_out = posEnc.bwd(dec_out[:, self.dec_overlap:, :], self.encoder_len)
        return dec_out


def calc(train_input, test_input, epochs=100, opt="adam", lr=0.2, seq_len=96, pred_len=96, enc_in=13, n_hidden_enc=256,
         n_hidden_dec=256, n_heads=4, dropout=0.1, dec_overlap=5, d_model=256):

    model = Transformer(forecasting_len=pred_len, encoder_len=seq_len, n_heads=n_heads, n_hidden_enc=n_hidden_enc,
                        n_hidden_dec=n_hidden_dec, dropout=dropout, dec_overlap=dec_overlap, d_model=d_model)

    if opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.LBFGS(model.parameters(), lr=lr)

    return utils_ba.optimize(optimizer=optimizer, test_input=test_input, train_input=train_input, epochs=epochs,
                             model=model, pred_len=pred_len, seq_len=seq_len)
