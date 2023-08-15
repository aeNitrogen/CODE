import torch

import models.PatchTST
import wandb

import data_loader
import patchAdapter
import numpy as np
import utils_ba
import torch.optim
import iterators.PatchTST_Iterator
import iterators.Autoformer_Iterator

import data_plotter


class training_iterator:
    def __init__(self, config: dict):
        print("DEBUG: iterator initialization")
        self.architecture = config["architecture"]

        self.batch_size = config["gpu_batch_size"]

        seed = config["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        print("DEBUG: iterator seeds set")

        self.train, self.val, _, _ = data_loader.load('SinData.pkl')  # change when running final experiments
        self.data_dim = self.train.size(2)
        config_translated = patchAdapter.translate_dict_actions(config, self.data_dim)

        print("DEBUG: config translated")
        assert config["kernel_size"] % 2 == 1, "please choose odd kernel_size"
        assert config["embed_type"] == 3 or config["embed_type"] == 4, "please choose another embed_type"
        self.overlap = config["overlap"]
        self.pred_len = config["prediction_length"]
        self.seq_len = config["lookback_window"]
        self.out_dim = config["real_output_size"]
        self.iterations = config["iterations"]
        self.iter = 0
        self.model = None

        if self.architecture == "transformer":
            print("TODO")
            # self.model = transformer.Transformer

        elif self.architecture == "PatchTST":
            self.model = patchAdapter.patchtst(config_translated)

        elif self.architecture == "Autoformer":
            self.model = patchAdapter.autoformer(config_translated)

        else:
            assert False, "please specify a supported architecture"

        print("DEBUG: model initialized")

        # print(config_translated)

        lr = config["learning_rate"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        print("DEBUG: iterator initialization finished")

    def iterate(self):
        batch_size = self.batch_size
        criterion = torch.nn.MSELoss()
        mae_crit = torch.nn.L1Loss()

        final = False

        if self.architecture == "PatchTST":

            self.model.train()

            iterators.PatchTST_Iterator.optimize(self.model, self.optimizer, self.train, batch_size, self.data_dim,
                                                 self.out_dim, criterion, self.pred_len, self.seq_len)

            self.iter += 1
            if self.iter == self.iterations:
                final = True

            self.model.eval()

            with torch.no_grad():
                ret_dict = iterators.PatchTST_Iterator.predict(self.model, self.train, batch_size, self.data_dim,
                                                               self.out_dim, criterion, mae_crit, self.pred_len,
                                                               self.seq_len, final=final)



        elif self.architecture == "Autoformer":

            self.model.train()

            iterators.Autoformer_Iterator.optimize(self.model, self.optimizer, self.train, batch_size, self.data_dim,
                                                   self.out_dim, criterion, self.pred_len, self.seq_len)

            self.model.eval()
            with torch.no_grad():
                x, y = utils_ba.get_single(self.val, self.pred_len, self.seq_len, self.data_dim, self.out_dim)

                # out = self.model.forward(x)
                out = utils_ba.split_prediction_autoformer(self.model, x, batch_size)

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

        return ret_dict

    def finalize(self):
        batch_size = self.batch_size
        truth = torch.squeeze(self.val[0, :, :])
        if self.architecture == "PatchTST":
            x, _ = utils_ba.get_single_pred(self.val, self.pred_len, self.seq_len, self.data_dim, self.out_dim)
            zeros = torch.zeros_like(self.val[0, :, :])
            with torch.no_grad():

                prediction = utils_ba.split_prediction_single(self.model, x, batch_size)

            prediction = prediction[0, :, :]
            zeros[self.seq_len: self.pred_len + self.seq_len, :] = prediction
            prediction = torch.squeeze(zeros)
            return data_plotter.log_pred_plots(prediction.cpu(), truth.cpu())

        elif self.architecture == "Autoformer":
            x, _ = utils_ba.get_single_pred(self.val, self.pred_len, self.seq_len, self.data_dim, self.out_dim)
            zeros = torch.zeros_like(self.val[0, :, :])
            with torch.no_grad():

                prediction = utils_ba.split_prediction_autoformer(self.model, x, batch_size)

            prediction = prediction[0, :, :]
            zeros[self.seq_len: self.pred_len + self.seq_len, :] = prediction
            prediction = torch.squeeze(zeros)
            return data_plotter.log_pred_plots(prediction.cpu(), truth.cpu())
        # wandb.log({"truth": truth, "prediction": prediction})

    def fin(self):
        print("max_iter: " + self.iter.__str__())
        print("out of: " + self.iterations.__str__())
