import functools

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

        # initialise the model parameters
        self.params = self._initialise_parameters()

        # initialise the optimiser
        try:
            optim_fn = getattr(optax, optim_name)
        except AttributeError:
            raise ValueError(f"optax has no optimiser called `{optim_name}`")
        self.optim = optim_fn(**optim_settings)
        self.opt_state = self.optim.init(self.params)

        # load the checkpoint
        self.load(checkpoint_id=checkpoint_id)

    def _initialise_parameters(self):
        # pass dummy data to setup the network parameters
        batch = ChessDataset.get_dummy_batch(batch_size=8)
        return self.forward.init(self.rng_key, is_training=True, **batch)

    @functools.partial(jax.jit, static_argnums=(0, 4))
    def compute_loss(self, params, rng, batch, is_training: bool = True):
        """
        Compute the loss for a batch.
        :param params:
        :param rng:
        :param batch:
        :param is_training:
        :return:
        """
        output = self.forward.apply(params, rng, is_training=is_training, **batch)
        target = batch["next_move"]
        mask = batch["legal_moves"]
        loss = cross_entropy(output, target, mask)
        loss = jnp.mean(loss, axis=0)
        return loss, output

    @functools.partial(jax.jit, static_argnums=(0, 4))
    def compute_grads(self, params, rng, batch, is_training: bool = True):
        """
        Compute the gradients for a batch.
        :param params:
        :param rng:
        :param batch:
        :param is_training:
        :return:
        """
        return jax.value_and_grad(self.compute_loss, has_aux=True)(params, rng, batch, is_training)

    @functools.partial(jax.jit, static_argnums=(0,))
    def optimiser_step(self, grads, opt_state, params):
        updates, opt_state = self.optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    def train(self, loader: Optional[DataLoader] = None):
        # grab the train loader
        if loader is None:
            assert self.loaders is not None, "No loaders provided."
            loader = self.loaders.train

        for i, batch in enumerate(tqdm(loader)):
            # forward pass + gradient computation
            key = self.next_rng_key()
            (loss, output), grads = self.compute_grads(self.params, key, batch, is_training=True)

            # update parameters
            self.params, self.opt_state = self.optimiser_step(grads, self.opt_state, self.params)

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

        losses = []
        summary_stats = {}
        for i, batch in enumerate(tqdm(loader)):
            # forward pass
            key = self.next_rng_key()  # note: not really needed since dropout is disabled?
            loss, _output = self.compute_loss(self.params, key, batch, is_training=False)

            # track metrics
            losses.append(loss.item())
            self._plot_loss(
                {
                    "loss": loss,
                },
                train=False,
            )

        return np.mean(losses), summary_stats

    def get_state_dict(self) -> Dict:
        return {
            "params": self.params,
            "optim_state": self.opt_state,
        }

    def load_state_dict(self, state_dict: Dict):
        self.params = state_dict["params"]
        self.opt_state = state_dict["optim_state"]
