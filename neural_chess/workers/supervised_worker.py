import haiku as hk
from hijax import AbstractWorker


class SupervisedWorker(AbstractWorker):
    def __init__(self, model, checkpoint_id="best", *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, loader):
        pass

    def evaluate(self, loader):
        pass
