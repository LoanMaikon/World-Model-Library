import os

class LeWorldModel():
    def __init__(self,
                 opened_config,
                 output_folder,
                 device,
                 rank,
                 world_size,
                 resume,
                 evaluate_state=None,
                ):

        self.config = opened_config
        self.output_folder = output_folder
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.resume = resume
        self.evaluate_state = evaluate_state
        self.is_evaluating = self.config["mode"] == "evaluate"

        self._load_config()

        # ...

    def train(self):
        pass

    def test(self):
        pass # calls self._evaluate() with the chosen solver

    def _evaluate(self):
        pass

    def _load_config(self):
        pass
