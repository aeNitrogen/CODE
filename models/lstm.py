import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        input_size = configs.enc_in
        n_layers = configs.e_layers
        self.proj_size = configs.c_out
        self.enc_len = configs.enc_len

        dropout = configs.dropout
        if configs.individual:
            n_hidden = configs.c_out
            self.lstm = nn.LSTM(input_size, n_hidden, num_layers=n_layers, batch_first=True, dropout=dropout)
        else:
            n_hidden = configs.d_model
            self.lstm = nn.LSTM(input_size, n_hidden, num_layers=n_layers, batch_first=True, dropout=dropout,
                                proj_size=self.proj_size)

    def forward(self, start_token, x):
        olde = False
        if olde:
            _, weights = self.lstm(start_token)
            out, _ = self.lstm(x, weights)
        else:
            in_s = torch.cat((start_token, x), dim=1)
            out, _ = self.lstm(in_s)
            out = out[:, -self.pred_len:, :]
        return out
