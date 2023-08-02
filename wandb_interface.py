import wandb


def log(data, seq_len, pred_len, opt, epochs, architecture, lr=0.2, info="", hidden_dec=-1, hidden_enc=-1, d_model=-1,
        dropout=0.0, overlap=-1, n_heads=-1, pred=None, target=None):
    WANDB_API_KEY = "0542663e58cbd656b41998c3db626e17e4276f16"
    # WANDB_NAME = "run"
    wandb.login(key=WANDB_API_KEY)
    # start a new wandb run to track this script

    name = architecture + "-lr:" + lr.__str__() + "-epochs:" + epochs.__str__() + "-pred:" + pred_len.__str__() + \
           '-seq:' + seq_len.__str__()

    wandb.init(
        # set the wandb project where this run will be logged
        project="battery",

        # track hyperparameters and run metadata
        config={
            "learning_rate": lr,
            "architecture": architecture,
            "dataset": "Cust_1",
            "epochs": epochs,
            "info": info,
            "optimizer": opt,
            "pred_len": pred_len,
            "seq_len": seq_len,
            "hidden_dec": hidden_dec,
            "hidden_enc": hidden_enc,
            "d_model": d_model,
            "dropout": dropout,
            "overlap": overlap,
            "n_heads": n_heads
        },

        name=name
    )

    if pred is not None and target is not None:
        stop = min(len(data), pred.size()[0])
        final = max(len(data), pred.size()[0])
        pred_longer = pred.size()[0] >= len(data)
        for g in range(stop):
            dic_t = {"loss": data[g]}
            for i in range(pred.size()[1]):
                dic_t["target " + i.__str__()] = target[g, i]
                dic_t["prediction " + i.__str__()] = pred[g, i]
            wandb.log(dic_t)
        if pred_longer:
            for g in range(final - stop):
                dic_t = {}
                for i in range(pred.size()[1]):
                    dic_t["target " + i.__str__()] = target[g + stop, i]
                    dic_t["prediction " + i.__str__()] = pred[g + stop, i]
                wandb.log(dic_t)
        else:
            for g in range(final - stop):
                wandb.log({"loss": data[g + stop]})
    else:
        for loss in data:
            wandb.log({"loss": loss})
    wandb.finish()

def getName(config: dict, iterations):

    name = config["architecture"] + "-lr:" + config["learning_rate"] + "-epochs:" + iterations + "-pred_len:" + \
           config["prediction_length"] + '-lbw:' + config["lookback_window"]
    return name


def init(config: dict, iterations):

    WANDB_API_KEY = "0542663e58cbd656b41998c3db626e17e4276f16"
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project="battery",
        config=config,
        name=getName(config, iterations)
        )


def log_dict(results: dict):
    wandb.log(results)

