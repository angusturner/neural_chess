from torch.utils.data import DataLoader
from typing import Optional, Callable

from pprint import pprint

import jax
import jax.numpy as jnp
import haiku as hk
import hijax as hx
from neural_chess.datasets import ChessDataset

from neural_chess.utils import cross_entropy


class SupervisedWorker(hx.Worker):
    def __init__(
        self,
        model: Callable,
        loaders: Optional[hx.Loaders] = None,
        checkpoint_id: str = "best",
        *args,
        **kwargs,
    ):
        super().__init__(loaders=loaders, *args, **kwargs)

        # transform the model into a pure jax function
        self.forward: hk.Transformed = hk.transform(model)
        self.rng = jax.random.PRNGKey(42)

        # initialise the parameters
        self.params = self._initialise_parameters()

        # register the parameters for model checkpointing
        self.register_state(self.params, "params")

        # load the checkpoint
        self.load(checkpoint_id=checkpoint_id)

    def _initialise_parameters(self):
        # pass dummy data to setup the network parameters
        batch = ChessDataset.get_dummy_batch()
        return self.forward.init(self.rng, is_training=True, **batch)

    def get_criterion(self) -> Callable:
        """
        Return a closure which:
        - uses the `forward.apply` function to compute the model predictions
        - computes the cross entropy loss
        - computes the gradients of the loss with respect to the model parameters
        """

        @jax.jit
        def compute_loss(params, rng, batch, is_training=True):
            output = self.forward.apply(params, rng, is_training=is_training, **batch)
            target = batch["next_move"]
            mask = batch["legal_moves"]
            loss = cross_entropy(output, target, mask)
            loss = jnp.mean(loss, axis=0)
            return loss, output

        return compute_loss

    def train(self, loader: Optional[DataLoader] = None):
        # grab the train loader
        if loader is None:
            assert self.loaders is not None, "No loaders provided."
            loader = self.loaders.train

        # <DEBUG>
        def check_grad(x: jnp.ndarray):
            if jnp.isnan(x).any() or jnp.isinf(x).any():
                raise ValueError("NaN or Inf detected")

        # </DEBUG>

        # get the criterion
        criterion = self.get_criterion()
        for i, batch in enumerate(loader):
            # forward pass
            (loss, output), grads = jax.value_and_grad(criterion, has_aux=True)(self.params, self.rng, batch)

            print("Gradients:")
            jax.tree_multimap(check_grad, grads)
            grad_shapes = jax.tree_multimap(lambda x: x.shape, grads)
            pprint(grad_shapes)

            print(f"Loss: {loss}")

            raise Exception("we made it here!")

    def evaluate(self, loader: Optional[DataLoader] = None):
        # grab the test loader
        if loader is None:
            assert self.loaders is not None, "No loaders provided."
            loader = self.loaders.test

        raise NotImplementedError("Evaluation not implemented yet.")
