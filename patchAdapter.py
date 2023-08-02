from argparse import Namespace
import torch.cuda

def translate_dict_actions(config: dict):
    args = Namespace(

        # Forecasting task
        seq_len = config["lookback_window"] + config["prediction_length"], # input sequence length
        label_len= config["overlap"] + config["prediction_length"], # start token length
        pred_len = config["prediction_length"],

        # Linear
        individual= config["individual"],

        #PatchTST
        fc_dropout=, # fully connected dropout
        head_dropout = , # head dropout
        patch_len = , # patch length
        stride = , # stride
        padding_patch = , # None: None; end: padding on the end
        revin = , # RevIN; True 1 False 0
        affine = , # RevIN-affine; True 1 False 0
        subtract_last = , # 0: subtract mean; 1: subtract last
        decomposition = , # decomposition; True 1 False 0
        kernel_size = , # decomposition-kernel
        e_layers = , # individual head; True 1 False 0

        # Formers
        embed_type = , # 0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding
        enc_in = config["lookback_window"],  # encoder input size
        dec_in = config["overlap"] + config["prediction_length"],
        c_out = , # output size
        d_model = config["d_model"], # model dimension
        n_heads = config["n_heads"], # number of heads
        e_layers = config["encoder_layers"], # num of encoder layers
        d_layers = config["decoder_layers"], # number of decoder layers
        d_ff = , # dimension of fcn
        moving_avg = 25, # window size moving average
        factor = , # attention factor
        distil = , # whether to use distilling in encoder, using this argument means not using distilling
        dropout=config["dropout"],
        embed = , # time features encoding, options:[timeF, fixed, learned]
        activation = , # activation function
        output_attention = , # wether to output attention in ecoder
        do_predict = , # whether to predict unseen future data

        # misc
        use_gpu = True if torch.cuda.is_available() else False,
    )


def patchtst(config: dict):
