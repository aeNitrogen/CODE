import torch

import transformer
import data_loader
import patchAdapter
import numpy as np
import utils_ba
import torch.optim


class training_iterator:
    def __init__(self, config: dict):
        self.architecture = config["architecture"]

        seed = config["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.train, self.val, _, _ = data_loader.load('SinData.pkl')  # change when running final experiments
        self.data_dim = self.train.size(2)
        config = patchAdapter.translate_dict_actions(config, self.data_dim)

        self.overlap = config["overlap"]
        self.pred_len = config["prediction_length"]
        self.seq_len = config["lookback_window"]
        self.out_dim = config["output_size"]

        self.model = None

        if self.architecture == "transformer":
            self.model = transformer.Transformer
        elif self.architecture == "PatchTST":
            self.model = patchAdapter.patchtst(config)
        else:
            assert False, "please specify a supported architecture"

        lr = config["learning_rate"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def iterate(self):
        if self.architecture == "PatchTST":
            self.model.forward()
            self.model.train()

            criterion = torch.nn.MSELoss()
            mae_crit = torch.nn.L1Loss()

            def closure():
                self.optimizer.zero_grad()
                x, y = utils_ba.get_single(self.train, self.pred_len, self.seq_len, self.data_dim, self.out_dim)
                loss = criterion(x, y)
                loss.backward()
                return loss

            self.optimizer.step(closure)

            ret_dict = {}

            self.model.eval()
            with torch.no_grad():
                x, y = utils_ba.get_single(self.val, self.pred_len, self.seq_len, self.data_dim, self.out_dim)
                loss = criterion(x, y)
                rmse = np.sqrt(loss)
                mae = mae_crit(x, y)
                rmae = np.sqrt(mae)
                ret_dict = {
                    'validation_loss': loss,
                    'rmse': rmse,
                    'mae': mae,
                    'rmae': rmae
                }
            return ret_dict
