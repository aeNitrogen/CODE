import models.s4.s4 as s4
from torch import nn
from torch import Tensor


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        enc_in = configs.enc_in
        context_window = configs.seq_len
        self.target_window = configs.pred_len
        c_out = configs.c_out

        self.n_layers = configs.n_layers

        d_model = configs.d_model
        bottleneck = configs.bottleneck

        gate = configs.gate
        gate_act = configs.gate_act
        mult_act = configs.mult_act
        final_act = configs.final_act
        dropout = configs.dropout
        tie_dropout = configs.tie_dropout
        transposed = False # configs.transposed

        d_state = configs.d_state
        channels = configs.channels


        # possible s4block kwargs:
        '''
        d_state: int
        deterministic: bool
        channels: channels
        
        
        taken from S4 kernel: 
        "d_model (H): Model dimension, or number of independent convolution kernels created.
        channels (C): Extra dimension in the returned output (see .forward()).
            - One interpretation is that it expands the input dimension giving it C separate "heads" per feature.
              That is convolving by this kernel maps shape (B L D) -> (B L C D)
            - This is also used to implement a particular form of bidirectionality in an efficient way.
            - In general for making a more powerful model, instead of increasing C
              it is recommended to set channels=1 and adjust H to control parameters instead.
        "
        
        '''

        self.S4 = nn.ModuleList()
        for i in range(self.n_layers):
             self.S4.append(s4.S4Block(d_model, bottleneck=bottleneck, gate=gate, gate_act=gate_act, mult_act=mult_act,
                             final_act=final_act, dropout=dropout, tie_dropout=tie_dropout,
                             transposed=transposed, d_state=d_state, channels=channels))

        self.encoder = nn.Linear(enc_in, d_model)

        self.decoder = nn.Linear(d_model, c_out)

    def forward(self, x):

        x_in = self.encoder(x)

        for i in range(self.n_layers):
            x_in, _ = self.S4[i](x_in)
        x_s4 = x_in

        x_restricted = x_s4[:, -self.target_window:, :]

        x_out = self.decoder(x_restricted)

        return x_out
