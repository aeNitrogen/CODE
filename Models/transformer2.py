import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn import functional as F


torch.manual_seed(8055)
native_batch_size = 750    # number of samples/batch size
native_train_batch = 500
batch_size = 30
block_size = 10     # maximum context length for predictions
L = 900  # length of each sample
T = 13  # input var count
Out_num = 9  # output var count
dType = torch.float32
np.random.seed(0)

trans = nn.Transformer(T, nhead=13, batch_first=True)



def calc(train_input, train_targets, test_input, test_targets):
    model = trans
    # print(model.parameters().__str__())
    criterion = nn.MSELoss()
    optimizer = optim.LBFGS(model.parameters(), lr=0.8)

    n_steps = 10
    for i in range(n_steps):
        print("step", i)

        def closure():
            optimizer.zero_grad()
            in_batch, in_targets = getMinibatch(train_input, train_targets)
            out = model(in_batch, in_targets)
            # print(out.size())
            loss = criterion(out, in_targets)
            print("loss", loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            in_batch, in_targets = getMinibatch(train_input, train_targets)
            pred = model(in_batch, in_targets)
            loss = criterion(in_batch, in_targets)
            print("test loss", loss.item())
