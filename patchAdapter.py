from argparse import Namespace
import torch.cuda
import models.PatchTST
import models.Autoformer
import models.Transformer
import models.Informer
import models.NuLinear
import models.NLinear
import models.DLinear
import models.NuLin
import models.lstm
import models.Informer_plus
import ns_models.ns_Informer
import ns_models.ns_Autoformer
import ns_models.ns_Transformer
import models.s4py

def set_attr(args: Namespace, config: dict, name: str):
    args.__setattr__(name, config[name])

def translate_dict_actions(config: dict, data_dim, architecture=""):
    args = Namespace(

        # Forecasting task
        seq_len=config["lookback_window"] + config["prediction_length"],  # input sequence length
        label_len=config["overlap"] + config["prediction_length"],  # start token length
        pred_len=config["prediction_length"],
        enc_len=config["lookback_window"],                          # encoder sequence length

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

        freq=config["frequency"],

        # Formers
        embed_type=config["embed_type"],
        # 0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal
        # embedding 3: value embedding + positional embedding 4: value embedding
        enc_in=data_dim,  # enc input data dimension (for embedding)
        dec_in=data_dim,  # dec input data dimension (for embedding)
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

    if architecture in ["Transformer", "Informer", "Informer+"]:
        args.__setattr__("dec_in", data_dim - config["output_size"])  # was 4
        print("DEBUG: dec_in = " + (data_dim - config["output_size"]).__str__())

    if architecture in ["NuLin", "NuLinear"]:
        args.__setattr__("seq_len", config["lookback_window"])  # TODO see input provider

    if architecture in ["NuLin"]:
        if config["revin"] == 1:
            args.__setattr__("action_size", 4)
            args.__setattr__("enc_in", config["output_size"]) # 9
            args.__setattr__("dec_in", data_dim - config["output_size"]) # 4
        else:
            args.__setattr__("action_size", config["output_size"])
            args.__setattr__("dec_in", config["output_size"])
            args.__setattr__("enc_in", data_dim - config["output_size"])

    if architecture in ["ns_Transformer", "ns_Informer", "ns_Autoformer"]:
        args.__setattr__("p_hidden_dims", config["p_hidden_dims"])
        args.__setattr__("p_hidden_layers", config["p_hidden_layers"])
        args.__setattr__("seq_len", config["lookback_window"])  # TODO see input provider
        # print("hidden layers set as " + config["p_hidden_layers"].__str__())
        # print(type(config["p_hidden_layers"]))

    if architecture in ["s4"]:
        set_attr(args, config, "n_layers")
        set_attr(args, config, "d_state")
        set_attr(args, config, "bottleneck")
        set_attr(args, config, "gate")
        set_attr(args, config, "gate_act")
        set_attr(args, config, "mult_act")
        set_attr(args, config, "final_act")
        set_attr(args, config, "tie_dropout")
        set_attr(args, config, "transposed")
        set_attr(args, config, "channels")

        if args.gate == 0:
            args.__setattr__("gate", None)
        if args.bottleneck == 0:
            args.__setattr__("bottleneck", None)
        if args.channels == 0:
            args.__setattr__("channels", None)


    return args


def patchtst(config: dict):
    model = models.PatchTST.Model(config)
    return model


def autoformer(config: dict):
    model = models.Autoformer.Model(config)
    return model


def transformer(config: dict):
    model = models.Transformer.Model(config)
    return model


def informer(config: dict):
    model = models.Informer.Model(config)
    return model


def nlinear(config: dict):
    model = models.NLinear.Model(config)
    return model


def dlinear(config: dict):
    model = models.DLinear.Model(config)
    return model


def nulinear(config: dict):
    model = models.NuLinear.Model(config)
    return model


def nulin(config: dict):
    model = models.NuLin.Model(config)
    return model


def lstm(config: dict):
    model = models.lstm.Model(config)
    return model


def informer_plus(config: dict):
    model = models.Informer_plus.Model(config)
    return model


def ns_informer(config: dict):
    model = ns_models.ns_Informer.Model(config)
    return model


def ns_transformer(config: dict):
    model = ns_models.ns_Transformer.Model(config)
    return model


def ns_autoformer(config: dict):
    model = ns_models.ns_Autoformer.Model(config)
    return model

def s4(config: dict):
    model = models.s4py.Model(config)
    return model
