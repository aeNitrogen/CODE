import transformer
import patchAdapter

class training_iterator:
    def __init__(self, config: dict):
        architecture = config["architecture"]
        self.model = None

        if architecture == "transformer":
            self.model = transformer.Transformer
        if architecture == "PatchTST":
            self.model = patchAdapter.patchtst(config)
