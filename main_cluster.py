from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging
from cw2.experiment import ExperimentSurrender
import ClusterAdapter
from sweep_work.create_sweep import create_sweep
from sweep_work.sweep_logger import SweepLogger
from sweep_work.experiment_wrappers import wrap_experiment
import wandb
import wandb_interface
from training_iterator import training_iterator
import torch


class HyperparamOpt_nonIterative(experiment.AbstractExperiment):

    def run(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        ClusterAdapter.run(cw_config)

    def initialize(
        self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray
    ) -> None:
        pass

    def finalize(self, surrender: ExperimentSurrender = None, crash: bool = False):
        pass

class HyperparamOpt(experiment.AbstractIterativeExperiment):
    def __init__(self):
        self.iterator = None

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        pass

    def initialize(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        print("DEBUG: Experiment initialization")
        wandb_interface.init(cw_config["params"], cw_config["iterations"])
        print("DEBUG: wandb interface initialized")
        with torch.device("cuda:0"):
            self.iterator = training_iterator(cw_config["params"])
        print("DEBUG: Experiment initialization completed")

    def finalize(self, surrender: ExperimentSurrender = None, crash: bool = False):
        pass

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:

        with torch.device("cuda:0"):
            return self.iterator.iterate()


if __name__ == "__main__":
    print("start_main")

    # WANDB_API_KEY = "0542663e58cbd656b41998c3db626e17e4276f16"
    # wandb.login(key=WANDB_API_KEY)  # , anonymous="never", )

    cw = cluster_work.ClusterWork(wrap_experiment(HyperparamOpt))
    create_sweep(cw)
    cw.add_logger(SweepLogger())
    cw.run()
