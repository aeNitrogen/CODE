import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import time


class Model(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, seq_len=96, pred_len=10, enc_in=13, individual=False):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = enc_in
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len + self.pred_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len + self.pred_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        if x.size(2) > Out_num:
            x = x[:, :, T - Out_num:]
        return x  # [Batch, Output length, Channel]

T = 13
Out_num = 9
def getMinibatch(input, fLen=10, iLen=96, info=False):
    length = input.size()[1]
    # num_batches = length / iLen
    start_pos = np.random.randint(0, length - iLen - fLen)
    input_batched = input[:, start_pos:start_pos + iLen, :]
    target = input[:, start_pos + iLen:start_pos + iLen + fLen, :]
    start_token = target.clone()
    if info:
        print("one")
        print(target[0, :, :])
    start_token[:, :, T - Out_num:] = torch.zeros_like(target[:, :, T - Out_num:])
    if info:
        print("two")
        print(target[0, :, :])
    input_batched = torch.cat((input_batched, start_token), 1)
    return input_batched, target


def calc(train_input, test_input, epochs=100, opt="adam", lr=0.2, seq_len=96, pred_len=96, enc_in=13,
         individual=False):
    loss_array = []
    model = Model(seq_len, pred_len, enc_in, individual)

    criterion = nn.MSELoss()
    if opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.LBFGS(model.parameters(), lr=lr)

    n_steps = epochs
    for i in range(n_steps):
        # print("step", i)
        # model.train_mode()

        def closure():
            start = time.time()
            optimizer.zero_grad()
            enc_in, target = getMinibatch(train_input, fLen=pred_len, iLen=seq_len)
            target = target[:, :, T - Out_num:]
            out = model(enc_in)
            loss = criterion(out, target)
            # print("loss", loss.item())
            loss.backward()
            # print(time.time() - start)
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            enc_in, target = getMinibatch(test_input, fLen=pred_len, iLen=seq_len)
            # model.eval_mode()
            pred = model(enc_in)
            target = target[:, :, T - Out_num:]
            loss = criterion(pred, target)
            loss_array.append(loss)
            # print("test loss", loss.item())
    with torch.no_grad():
        enc_in, target = getMinibatch(test_input, fLen=pred_len, iLen=seq_len, info=False)
        # model.eval_mode()
        pred = model(enc_in)
        target = target[:, :, T - Out_num:]
    print('loss: ' + loss_array[-1:][0].__str__())
    return loss_array, pred[0, :, :], target[0, :, :]
