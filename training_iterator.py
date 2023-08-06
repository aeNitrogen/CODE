import torch

import transformer
import data_loader
import patchAdapter
import numpy as np
import utils_ba
import torch.optim


class training_iterator:
    def __init__(self, config: dict):
        print("DEBUG: iterator initialization")
        self.architecture = config["architecture"]

        seed = config["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        print("DEBUG: iterator seeds set")

        self.train, self.val, _, _ = data_loader.load('SinData.pkl')  # change when running final experiments
        self.data_dim = self.train.size(2)
        config_translated = patchAdapter.translate_dict_actions(config, self.data_dim)

        print("DEBUG: config translated")

        self.overlap = config["overlap"]
        self.pred_len = config["prediction_length"]
        self.seq_len = config["lookback_window"]
        self.out_dim = config["output_size"]

        self.model = None

        if self.architecture == "transformer":
            print("TODO")
            self.model = transformer.Transformer

        elif self.architecture == "PatchTST":
            self.model = patchAdapter.patchtst(config_translated)
        else:
            assert False, "please specify a supported architecture"

        print("DEBUG: model initialized")

        lr = config["learning_rate"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        print("DEBUG: iterator initialization finished")

    def iterate(self):
        if self.architecture == "PatchTST":
            self.model.train()

            criterion = torch.nn.MSELoss()
            mae_crit = torch.nn.L1Loss()

            def closure():
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                x, y = utils_ba.get_single(self.train, self.pred_len, self.seq_len, self.data_dim, self.out_dim)
                out = self.model.forward(x)
                # print(out.size())
                out = out[:, :, self.data_dim - self.out_dim:]
                loss = criterion(out, y)
                print("here?")
                loss.backward()
                print("no")
                return loss

            self.optimizer.step(closure)
            # print("2")
            ret_dict = {}

            self.model.eval()
            with torch.no_grad():
                x, y = utils_ba.get_single(self.val, self.pred_len, self.seq_len, self.data_dim, self.out_dim)
                # print("bef")
                out = self.model.forward(x)
                # print("aft")
                out = out[:, :, self.data_dim - self.out_dim:]
                loss = criterion(out, y)
                # print("crit1")
                rmse = torch.sqrt(loss)
                # print("sqrt")
                mae = mae_crit(out, y)
                # print("crit2")
                rmae = torch.sqrt(mae)
                ret_dict = {
                    'validation_loss': loss,
                    'rmse': rmse,
                    'mae': mae,
                    'rmae': rmae
                }
            print("iter done")
        return ret_dict
