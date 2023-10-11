import torch
import data_loader
import patchAdapter
import numpy as np
import torch.optim
import iterators.PatchTST_Iterator
import iterators.Autoformer_Iterator
import iterators.Transformer_Iterator
import iterators.lstm_iterator


class training_iterator:
    def __init__(self, config: dict):
        print("DEBUG: iterator initialization")
        self.dataset = config["dataset"]
        self.architecture = config["architecture"]

        self.batch_size = config["gpu_batch_size"]

        seed = config["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        print("DEBUG: iterator seeds set")
        self.train, self.val, _, _ = data_loader.load(config["dataset"])  # change when running final experiments

        if config["final"]:
            print("DEBUG: final train set")
            _, _, self.train, self.val = data_loader.load(config["dataset"])  # for final runs

        self.data_dim = self.train.size(2)
        config_translated = patchAdapter.translate_dict_actions(config, self.data_dim, self.architecture)
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
        self.attn = config["output_attention"]
        print("DEBUG: ouput attention?: " + self.attn.__str__())
        lr = config["learning_rate"]

        if "ablation" in list(config.keys()):
            if config["ablation"] == "action_zero":
                self.train[:, :, :self.data_dim - self.out_dim] = torch.zeros_like(self.train[:, :, :self.data_dim - self.out_dim])
                self.val[:, :, :self.data_dim - self.out_dim] = torch.zeros_like(self.val[:, :, :self.data_dim - self.out_dim])
                print("actions set to zero")

        if self.architecture == "Transformer":
            self.model = patchAdapter.transformer(config_translated)

        elif self.architecture == "Informer+":
            self.model = patchAdapter.informer_plus(config_translated)

        elif self.architecture == "Lstm":
            self.model = patchAdapter.lstm(config_translated)

        elif self.architecture == "PatchTST":
            self.model = patchAdapter.patchtst(config_translated)

        elif self.architecture == "Autoformer":
            self.model = patchAdapter.autoformer(config_translated)

        elif self.architecture == "Informer":
            self.model = patchAdapter.informer(config_translated)

        elif self.architecture == "NLinear":
            self.model = patchAdapter.nlinear(config_translated)

        elif self.architecture == "DLinear":
            self.model = patchAdapter.dlinear(config_translated)

        elif self.architecture == "NuLinear":
            self.model = patchAdapter.nulinear(config_translated)

        elif self.architecture == "ns_Informer":
            self.model = patchAdapter.ns_informer(config_translated)

        elif self.architecture == "ns_Autoformer":
            self.model = patchAdapter.ns_autoformer(config_translated)

        elif self.architecture == "ns_Transformer":
            self.model = patchAdapter.ns_transformer(config_translated)

        else:
            assert False, "please specify a supported architecture"

        print("DEBUG: model initialized")

        # print(config_translated)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        print("DEBUG: iterator initialization finished")

    def iterate(self):
        batch_size = self.batch_size
        criterion = torch.nn.MSELoss()
        mae_crit = torch.nn.L1Loss()

        final = False

        self.iter += 1
        if self.iter == self.iterations:
            final = True

        if self.architecture in ["PatchTST", "NLinear", "DLinear"]:

            self.model.train()

            iterators.PatchTST_Iterator.optimize(self.model, self.optimizer, self.train, batch_size, self.data_dim,
                                                 self.out_dim, criterion, self.pred_len, self.seq_len)

            self.model.eval()

            with torch.no_grad():
                ret_dict = iterators.PatchTST_Iterator.predict(self.model, self.val, self.train, batch_size,
                                                               self.data_dim,
                                                               self.out_dim, criterion, mae_crit, self.pred_len,
                                                               self.seq_len, final=final, name=self.dataset)

        elif self.architecture in ["Lstm"]:

            self.model.train()

            iterators.lstm_iterator.optimize(self.model, self.optimizer, self.train, batch_size, self.data_dim,
                                             self.out_dim, criterion, self.pred_len, self.seq_len)

            self.model.eval()

            with torch.no_grad():
                ret_dict = iterators.lstm_iterator.predict(self.model, self.val, self.train, batch_size, self.data_dim,
                                                           self.out_dim, criterion, mae_crit, self.pred_len,
                                                           self.seq_len, final=final, name=self.dataset)

        elif self.architecture in ["ns_Transformer", "ns_Informer", "Transformer", "Informer", "NuLinear", "Informer+"]:
            self.model.train()

            iterators.Transformer_Iterator.optimize(self.model, self.optimizer, self.train, batch_size, self.data_dim,
                                                    self.out_dim, criterion, self.pred_len, self.seq_len,
                                                    attn=self.attn)

            self.model.eval()

            with torch.no_grad():
                ret_dict = iterators.Transformer_Iterator.predict(self.model, self.val, self.train, batch_size,
                                                                  self.data_dim, self.out_dim, criterion, mae_crit,
                                                                  self.pred_len, self.seq_len, final=final,
                                                                  attn=self.attn, name=self.dataset)

        elif self.architecture in ["Autoformer", "ns_Autoformer"]:

            self.model.train()

            iterators.Autoformer_Iterator.optimize(self.model, self.optimizer, self.train, batch_size, self.data_dim,
                                                   self.out_dim, criterion, self.pred_len, self.seq_len)

            self.model.eval()

            with torch.no_grad():
                ret_dict = iterators.Autoformer_Iterator.predict(self.model, self.val, self.train, batch_size,
                                                                 self.data_dim,
                                                                 self.out_dim, criterion, mae_crit, self.pred_len,
                                                                 self.seq_len, final=final, name=self.dataset)

        return ret_dict

    def fin(self):
        print("max_iter: " + self.iter.__str__())
        print("out of: " + self.iterations.__str__())
