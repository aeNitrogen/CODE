import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import utils_ba

N = 750  # number of samples
L = 900  # length of each sample
T = 13  # input var count
Out_num = 9  # output var count
dType = torch.float32
torch.manual_seed(8055)


def getMinibatch(input, fLen=96, iLen=96):
    length = input.size()[1]
    # num_batches = length / iLen
    start_pos = np.random.randint(0, length - iLen - fLen)
    input_batched = input[:, start_pos:start_pos + iLen, :]
    target = input[:, start_pos + iLen:start_pos + iLen + fLen, :]
    return input_batched, target[:, :, T - Out_num:], target[:, :, :T - Out_num]


class LSTM_pred(nn.Module):

    def __init__(self, d_model=10, forecasting_len=96, encoder_len=96):
        super(LSTM_pred, self).__init__()
        self.n_hidden = d_model
        self.pred_len = forecasting_len
        # lstm1, lstm2, ... , linear
        self.lstm1 = nn.LSTMCell(T, self.n_hidden)
        self.lstm2 = nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.linear = nn.Linear(self.n_hidden, Out_num)

    def eval_mode(self):
        self.linear.eval()
        self.lstm1.eval()
        self.lstm2.eval()

    def train_mode(self):
        self.linear.train()
        self.lstm1.train()
        self.lstm2.train()

    def forward(self, x, target_act):     # iterate through time steps; x should be of form x = [batch, features]
        outputs = []
        n_samples = x.size(0)  # number of samples (batch first)
        # n_samples replace by T? should be number of features, why is it number of samples????
        h_t1 = torch.zeros(n_samples, self.n_hidden, dtype=dType)  # initial hidden state
        c_t1 = torch.zeros(n_samples, self.n_hidden, dtype=dType)  # initial cell state
        h_t2 = torch.zeros(n_samples, self.n_hidden, dtype=dType)  # initial hidden state
        c_t2 = torch.zeros(n_samples, self.n_hidden, dtype=dType)  # initial cell state

        for input_t in x.split(1, dim=1):
            h_t1, c_t1 = self.lstm1(input_t.squeeze(), (h_t1, c_t1))  # calculate lstm layers and linear layer
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            output = self.linear(h_t2)

        for i in range(target_act.size(1)):     # calculate future predictions
            input = torch.zeros_like(x[:, 0, :])
            input[:, :T - Out_num] = target_act[:, i, :]
            input[:, T - Out_num:] = output
            h_t1, c_t1 = self.lstm1(input, (h_t1, c_t1))
            h_t2, c_t2 = self.lstm2(h_t1, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs.append(output)

        # outputs = torch.cat(outputs, dim=1)
        outputs = torch.stack(outputs, 1)
        return outputs


def calc(train_input, test_input, lr=0.2, d_model=128, epochs=100, pred_len=96,
         seq_len=96):
    model = LSTM_pred(d_model=d_model, forecasting_len=pred_len, encoder_len=seq_len)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    return utils_ba.optimize(optimizer=optimizer, test_input=test_input, train_input=train_input, epochs=epochs,
                             model=model, pred_len=pred_len, seq_len=seq_len, steps=True)
