import cw2.cluster_work
from cw2 import experiment, cw_error
from cw2.cw_data import cw_logging
from cw2.experiment import ExperimentSurrender
import ClusterAdapter

class HyperparamOpt(experiment.AbstractExperiment):

    def run(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        ClusterAdapter.run(cw_config)

    def initialize(
        self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray
    ) -> None:
        pass

    def finalize(self, surrender: ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == "__main__":
    cw = cw2.cluster_work.ClusterWork(HyperparamOpt)
    cw.run()
