import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Callable, Dict

import jax
import jax.numpy as jnp
import haiku as hk
import hijax as hx

import optax


from neural_chess.datasets import ChessDataset

from neural_chess.utils import cross_entropy


def check_grad(x: jnp.ndarray):
    if jnp.isnan(x).any() or jnp.isinf(x).any():
        raise ValueError("NaN or Inf detected")


class SupervisedWorker(hx.Worker):
    def __init__(
        self,
        model: Callable,
        loaders: Optional[hx.Loaders] = None,
        optim_name: str = "adam",
        optim_settings: Optional[Dict] = None,
        checkpoint_id: str = "best",
        *args,
        **kwargs,
    ):
        super().__init__(loaders=loaders, *args, **kwargs)
        if optim_settings is None:
            optim_settings = {}

        # transform the model into a pure jax function
        self.forward: hk.Transformed = hk.transform(model)
        self.rng = jax.random.PRNGKey(42)

        # initialise the model parameters
        self.params = self._initialise_parameters()

        # initialise the optimiser
        try:
            optim_fn = getattr(optax, optim_name)
        except AttributeError:
            raise ValueError(f"optax has no optimiser called `{optim_name}`")
        self.optim = optim_fn(**optim_settings)
        self.opt_state = self.optim.init(self.params)

        # register the parameters and optimiser state, for model checkpointing
        self.register_state(self.params, "params")
        self.register_state(self.opt_state, "optim_state")

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

        # get the criterion
        criterion = self.get_criterion()

        # define optimisation step
        @jax.jit
        def step(params, opt_state, batch):
            (loss, output), grads = jax.value_and_grad(criterion, has_aux=True)(params, self.rng, batch)
            updates, opt_state = self.optim.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return loss, output, params, opt_state

        for i, batch in enumerate(tqdm(loader)):
            # forward pass
            loss, output, self.params, self.opt_state = step(self.params, self.opt_state, batch)

            # plot metrics
            self._plot_loss(
                {
                    "loss": loss,
                },
                train=True,
            )

    def evaluate(self, loader: Optional[DataLoader] = None) -> (float, Dict):
        # grab the test loader
        if loader is None:
            assert self.loaders is not None, "No loaders provided."
            loader = self.loaders.test

        criterion = self.get_criterion()
        losses = []
        summary_stats = {}
        for i, batch in enumerate(tqdm(loader)):
            # forward pass
            eval_fn = jax.jit(jax.value_and_grad(criterion, has_aux=True))
            loss, _output = eval_fn(self.params, self.rng, batch, is_training=False)

            # track metrics
            losses.append(loss.item())
            self._plot_loss(
                {
                    "loss": loss,
                },
                train=False,
            )

        return np.mean(losses), summary_stats
