from argparse import Namespace
import torch.cuda
import models.PatchTST


def translate_dict_actions(config: dict, data_dim):
    args = Namespace(

        # Forecasting task
        seq_len=config["lookback_window"] + config["prediction_length"],  # input sequence length
        label_len=config["overlap"] + config["prediction_length"],  # start token length
        pred_len=config["prediction_length"],

        # PatchTST
        fc_dropout=config["fc_dropout"],  # fully connected dropout
        head_dropout=config["head_dropout"],  # head dropout
        patch_len=config["patch_length"],  # patch length
        stride=config["stride"],  # stride
        padding_patch=config["padding_patch"],  # None: None; end: padding on the end
        revin=config["revin"],  # RevIN; True 1 False 0
        affine=config["affine"],  # RevIN-affine; True 1 False 0
        subtract_last=config["subtract_last"],  # 0: subtract mean; 1: subtract last
        decomposition=config["decomposition"],  # decomposition; True 1 False 0
        kernel_size=config["kernel_size"],  # decomposition-kernel
        individual=config["individual"],  # individual head; True 1 False 0

        # Formers
        embed_type=0,
        # 0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal
        # embedding 3: value embedding + positional embedding 4: value embedding
        enc_in=data_dim,  # enc input data dimension (for embedding)
        dec_in=data_dim,
        c_out=config["output_size"],  # output size
        d_model=config["d_model"],  # model dimension
        n_heads=config["n_heads"],  # number of heads
        e_layers=config["encoder_layers"],  # num of encoder layers
        d_layers=config["decoder_layers"],  # number of decoder layers
        d_ff=config["d_fcn"],  # dimension of fcn
        moving_avg=config["moving_average"],  # window size moving average
        factor=1,  # attention factor
        distil=config["distilling"],
        # whether to use distilling in encoder, using this argument means not using distilling
        dropout=config["dropout"],
        embed=config["embed"],  # time features encoding, options:[timeF, fixed, learned]
        activation=config["activation"],  # activation function
        output_attention=config["output_attention"],  # whether to output attention in encoder
        # do_predict = , # whether to predict unseen future data

        # misc
        use_gpu=True if torch.cuda.is_available() else False,
    )

    return args


def patchtst(config: dict):
    model = models.PatchTST.Model(config)
    return model
