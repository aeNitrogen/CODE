import torch
import torch.nn as nn
import math
from torch import nn, Tensor


class PositionalEncoder(nn.Module):
    """
    The authors of the original transformer paper describe very succinctly what
    the positional encoding layer does and why it is needed:

    "Since our model contains no recurrence and no convolution, in order for the
    model to make use of the order of the sequence, we must inject some
    information about the relative or absolute position of the tokens in the
    sequence." (Vaswani et al, 2017)
    Adapted from:
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(  # dmodel has to be an even number
            self,
            dropout: float = 0.1,
            max_seq_len: int = 5000,
            d_model: int = 14,
            batch_first: bool = False
    ):
        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model
                     (Vaswani et al, 2017)
        """

        super().__init__()

        self.d_model = d_model

        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        self.x_dim = 1 if batch_first else 0

        # copy pasted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_len, 1, d_model)
        # pe2 = torch.zeros(1, max_seq_len, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        # pe2[0, :, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe2[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        # self.register_buffer('pe2', pe2)

    def forward(self, x: Tensor, dropout=True) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or
               [enc_seq_len, batch_size, dim_val]
        """
        pe_t = self.pe[:, :, :-1]  # only use if dmodel had to be rounded up
        # print(pe.size())
        # print(x.size())
        pe_t = pe_t[:x.size(self.x_dim)]
        pe_t = torch.swapaxes(pe_t, 0, 1)
        # print(pe.size())
        x = x + pe_t[:x.size(self.x_dim)]
        if not dropout:
            return x
        return self.dropout(x)

    def backward(self, x: Tensor, pos) -> Tensor:
        # diff = self.pe.size(1) - x.size(1)
        pe_t = self.pe[:, :, :x.size(2)]  # only use if dmodel had to be rounded up
        # print(pe_t.size(2))
        pe_t = pe_t[pos:x.size(self.x_dim) + pos]
        pe_t = torch.swapaxes(pe_t, 0, 1)
        x = x - pe_t[:x.size(self.x_dim)]
        return x

def fwd(x, dr):
    model = PositionalEncoder(batch_first=True)
    return model(x, dropout=dr)

def bwd(x, pos):
    model = PositionalEncoder(batch_first=True)
    return model.backward(x, pos)
