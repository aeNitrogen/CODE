# utility class
import torch.nn as nn
import numpy as np
import torch


def getMinibatch(input, fLen=96, iLen=96, T=13, Out_num=9):
    length = input.size()[1]
    start_pos = np.random.randint(0, length - iLen - fLen - 1)
    input_batched = input[:, start_pos:start_pos + iLen, :]
    target = input[:, start_pos + iLen:start_pos + iLen + fLen, :]
    return input_batched, target[:, :, T - Out_num:], target[:, :, :T - Out_num]


# _______actually used_______
def get_single(input, pred_len, seq_len, in_dim, out_dim):
    device = "cuda:0"
    length = input.size()[1]
    start_pos = np.random.randint(0, length - pred_len - seq_len - 1)
    input_batched = input[:, start_pos:start_pos + seq_len, :]  # slice the used input
    target = input[:, start_pos + seq_len:start_pos + seq_len + pred_len, :]  # whole target, including actions
    start_token = target.clone()
    start_token[:, :, in_dim - out_dim:] = torch.zeros_like(target[:, :, in_dim - out_dim:])
    input_batched = torch.cat((input_batched, start_token), 1)

    target_wo_actions = target[:, :, in_dim - out_dim:]

    assert input_batched.size(1) == seq_len + pred_len, "input size wrong"
    assert target_wo_actions.size(1) == pred_len, "target size wrong"

    return input_batched.to(device=device, copy=True), target_wo_actions.to(device=device, copy=True)


def get_single_pred(input, pred_len, seq_len, in_dim, out_dim):
    device = "cuda:0"
    length = input.size()[1]
    start_pos = 0
    input_batched = input[:, start_pos:start_pos + seq_len, :]  # slice the used input
    target = input[:, start_pos + seq_len:start_pos + seq_len + pred_len, :]  # whole target, including actions
    start_token = target.clone()
    start_token[:, :, in_dim - out_dim:] = torch.zeros_like(target[:, :, in_dim - out_dim:])
    input_batched = torch.cat((input_batched, start_token), 1)

    target_wo_actions = target[:, :, in_dim - out_dim:]

    assert input_batched.size(1) == seq_len + pred_len, "input size wrong"
    assert target_wo_actions.size(1) == pred_len, "target size wrong"

    return input_batched.to(device=device, copy=True), target_wo_actions.to(device=device, copy=True)


def getMinibatch_linear(input, fLen=10, iLen=96, T=13, Out_num=9, inf=False):
    length = input.size()[1]
    # num_batches = length / iLen
    start_pos = np.random.randint(0, length - iLen - fLen)
    input_batched = input[:, start_pos:start_pos + iLen, :]
    target = input[:, start_pos + iLen:start_pos + iLen + fLen, :]
    start_token = target.clone()
    if not inf:
        start_token[:, :, T - Out_num:] = torch.zeros_like(target[:, :, T - Out_num:])
    else:
        start_token[:, :, T - Out_num:] = torch.full_like(target[:, :, T - Out_num:], float('-inf'))
    input_batched = torch.cat((input_batched, start_token), 1)
    return input_batched, target


def optimize(optimizer, model, epochs, train_input, test_input, pred_len, seq_len, has_modes=True, linear=False,
             steps=False):
    criterion = nn.MSELoss()
    loss_array = []
    n_steps = epochs
    T = 13
    Out_num = 9
    for i in range(n_steps):
        if steps:
            print("step", i)
        if has_modes:
            model.train_mode()

        def closure():
            optimizer.zero_grad()
            if linear:
                enc_in, target = getMinibatch_linear(train_input, fLen=pred_len, iLen=seq_len)
                out = model(enc_in)
                target = target[:, :, T - Out_num:]
            else:
                enc_in, target, dec_in = getMinibatch(train_input, fLen=pred_len, iLen=seq_len)
                out = model(enc_in, dec_in)

            loss = criterion(out, target)
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad():

            if has_modes:
                model.eval_mode()
            if linear:
                enc_in, target = getMinibatch_linear(test_input, fLen=pred_len, iLen=seq_len)
                pred = model(enc_in)
                target = target[:, :, T - Out_num:]
            else:
                enc_in, target, dec_in = getMinibatch(test_input, fLen=pred_len, iLen=seq_len)
                pred = model(enc_in, dec_in)

            loss = criterion(pred, target)
            loss_array.append(loss.item())
            print("test loss: " + loss.item().__str__())
    with torch.no_grad():
        if linear:
            enc_in, target = getMinibatch_linear(test_input, fLen=pred_len, iLen=seq_len)
            pred = model(enc_in)
            target = target[:, :, 4:]
        else:
            enc_in, target, dec_in = getMinibatch(test_input, fLen=pred_len, iLen=seq_len)
            pred = model(enc_in, dec_in)
        # print('loss: ' + loss_array[-1:][0].__str__())
        print("final test loss", loss.item().__str__())

    return loss_array, pred[0, :, :], target[0, :, :]


def split_prediction_single(model, x, batch_size):
    assert x.size(0) % batch_size == 0, "please choose a batch size that is a divisor of the total batch number"
    y = model.forward(x[:batch_size, :, :])
    for i in range((x.size(0) // batch_size) - 1):
        j = i + 1
        y = torch.cat((y, model.forward(x[batch_size * j: batch_size * (j + 1), :, :])))
    return y


def split_prediction_autoformer(model, x, batch_size):
    assert x.size(0) % batch_size == 0, "please choose a batch size that is a divisor of the total batch number"
    zeros = torch.zeros_like(x[0:batch_size, :, :])

    y = model.forward(x[:batch_size, :, :], None, zeros, None)

    for i in range((x.size(0) // batch_size) - 1):
        j = i + 1
        y = torch.cat((y, model.forward(x[batch_size * j: batch_size * (j + 1), :, :], None, zeros, None)))
    return y
