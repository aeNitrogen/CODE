import torch
import wandb

import transformer
import data_loader
import patchAdapter
import numpy as np
import utils_ba
import torch.optim

import data_plotter


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
        self.iterations = config["iterations"]
        self.iter = 0
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
                out = out[:, :, self.data_dim - self.out_dim:]
                loss = criterion(out, y)
                loss.backward()
                return loss

            self.optimizer.step(closure)

            ret_dict = {}

            self.model.eval()
            with torch.no_grad():
                x, y = utils_ba.get_single(self.val, self.pred_len, self.seq_len, self.data_dim, self.out_dim)

                out = self.model.forward(x)

                out = out[:, :, self.data_dim - self.out_dim:]
                loss = criterion(out, y)

                rmse = torch.sqrt(loss)

                mae = mae_crit(out, y)

                rmae = torch.sqrt(mae)
                ret_dict = {
                    'validation_loss': loss,
                    'rmse': rmse,
                    'mae': mae,
                    'rmae': rmae
                }
            self.iter += 1
            if self.iter == self.iterations:
                ret_dict.update(self.finalize())
                print(ret_dict)

        return ret_dict

    def finalize(self):
        truth = torch.squeeze(self.val[0, :, :])
        if self.architecture == "PatchTST":
            x, _ = utils_ba.get_single_pred(self.val, self.pred_len, self.seq_len, self.data_dim, self.out_dim)
            zeros = torch.zeros_like(self.val[0, :, :])
            with torch.no_grad():
                prediction = self.model.forward(x)
            prediction = prediction[0, :, :]
            zeros[self.seq_len: self.pred_len + self.seq_len, :] = prediction
            prediction = torch.squeeze(zeros)
            return data_plotter.log_pred_plots(prediction.cpu(), truth.cpu())

            #wandb.log({"truth": truth, "prediction": prediction})

    def fin(self):
        print("max_iter: " + self.iter.__str__())
        print("out of: " + self.iterations.__str__())
