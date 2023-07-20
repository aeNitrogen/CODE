import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

N = 750  # number of samples
L = 900  # length of each sample
T = 13  # input var count
Out_num = 9  # output var count
dType = torch.float32

class LSTM_pred(nn.Module):

    def __init__(self, n_hidden=10, lstmNr=2):
        super(LSTM_pred, self).__init__()
        self.n_hidden = n_hidden
        # lstm1, lstm2, ... , linear
        self.lstms = []
        self.lstms.append(nn.LSTMCell(T, self.n_hidden))
        for i in range(lstmNr - 1):
            self.lstms.append(nn.LSTMCell(self.n_hidden, self.n_hidden))
        self.linear = nn.Linear(self.n_hidden, Out_num)

    def forward(self, x, future=0):     # iterate through time steps; x should be of form x = [batch, features]
        outputs = []
        n_samples = x.size(0)  # number of samples (batch first)
        h_t = []
        c_t = []
        for i in range(len(self.lstms)):
            c_t.append(torch.zeros(n_samples, self.n_hidden, dtype=dType))
            h_t.append(torch.zeros(n_samples, self.n_hidden, dtype=dType))

        for input_t in x.split(1, dim=1):  # calculate lstm layers and linear layer
            h_t[0], c_t[0] = self.lstms[0](input_t.squeeze(), (h_t[0], c_t[0]))
            for i in range(len(self.lstms) - 1):
                h_t[i + 1], c_t[i + 1] = self.lstms[i + 1](h_t[i], (h_t[i + 1], c_t[i + 1]))
            output = self.linear(h_t[len(self.lstms) - 1])
            outputs.append(output)

        for i in range(future):     # calculate future predictions
            h_t[0], c_t[0] = self.lstms[0](output, (h_t[0], c_t[0]))
            for i in range(len(self.lstms) - 1):
                h_t[i + 1], c_t[i + 1] = self.lstms[i + 1](h_t[i], (h_t[i + 1], c_t[i + 1]))
            output = self.linear(h_t[len(self.lstms) - 1])
            outputs.append(output)

        outputs = torch.stack(outputs, 1)
        return outputs


def calc(train_input, train_targets, test_input, test_targets):
    model = LSTM_pred()
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.8)

    n_steps = 10
    for i in range(n_steps):
        print("step", i)

        def closure():
            optimizer.zero_grad()
            out = model(train_input)
            loss = criterion(out, train_targets)
            print("loss", loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            future = 0
            pred = model(test_input, future=future)
            loss = criterion(pred, test_targets)
            print("test loss", loss.item())
