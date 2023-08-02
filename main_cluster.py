import cw2.cluster_work
from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging
from cw2.experiment import ExperimentSurrender
import ClusterAdapter
from sweep_work.create_sweep import create_sweep
from sweep_work.sweep_logger import SweepLogger
from sweep_work.experiment_wrappers import wrap_experiment
import wandb_interface
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
        wandb_interface.init(cw_config["parameters"], cw_config["iterations"])

    def finalize(self, surrender: ExperimentSurrender = None, crash: bool = False):
        # TODO: log prediction for visualization
        pass

    def iterate(self, cw_config: dict, rep: int, n: int) -> dict:
        return self.iterator.iterate()
        ClusterAdapter.run(cw_config)


if __name__ == "__main__":

    cw = cw2.cluster_work.ClusterWork(HyperparamOpt_nonIterative)

    # cw = cluster_work.ClusterWork(wrap_experiment(HyperparamOpt))

    create_sweep(cw)
    cw.add_logger(SweepLogger())

    cw.run()
