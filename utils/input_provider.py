import torch
import numpy as np


def get_input(input, pred_len, seq_len, in_dim, out_dim, prediction=False, two_inputs=False):
    """
    :param input:       input data
    :param pred_len:    prediction length
    :param seq_len:     lookback window
    :param in_dim:      input dimension
    :param out_dim:     output dimension
    :param prediction:  if prediction is to be made, start at index 0 to make graphs comparable
    :param two_inputs:  specifies wether there are two seperate inputs
    :return:            if two_inputs: encoder_input, decoder_input, target; else without decoder input (enc contains actions)
    """
    device = "cuda:0"
    length = input.size()[1]

    if prediction or length - pred_len - seq_len <= 1:
        start_pos = 0
    else:
        start_pos = np.random.randint(0, length - pred_len - seq_len - 1)

    input_batched = input[:, start_pos:start_pos + seq_len, :]  # slice the used input
    target = input[:, start_pos + seq_len:start_pos + seq_len + pred_len, :]  # whole target, including actions
    start_token = target.clone()
    start_token[:, :, in_dim - out_dim:] = torch.zeros_like(target[:, :, in_dim - out_dim:])

    target_wo_actions = target[:, :, in_dim - out_dim:]

    if two_inputs:
        start_token = start_token[:, :, 0:-out_dim]
        return input_batched.to(device=device, copy=True), start_token.to(device=device, copy=True),\
            target_wo_actions.to(device=device, copy=True)
    else:
        input_batched = torch.cat((input_batched, start_token), 1)
        return input_batched.to(device=device, copy=True), target_wo_actions.to(device=device, copy=True)