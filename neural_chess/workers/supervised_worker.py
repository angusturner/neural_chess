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


def get_accuracy(output, next_move):
    """
    Compute the accuracy of the model.
    :param output:
    :param next_move:
    :return:
    """
    # convert one-hot encoded next_move to index, and then compute accuracy
    next_move_idx = jnp.argmax(next_move, axis=1)
    correct = jnp.sum(jnp.equal(jnp.argmax(output, axis=1), next_move_idx))
    return correct / len(next_move_idx)


class SupervisedWorker(hx.Worker):
    def __init__(
        self,
        model: Callable,
        loaders: Optional[hx.Loaders] = None,
        optim_name: str = "adam",
        optim_settings: Optional[Dict] = None,
        checkpoint_id: str = "best",
        mask_loss: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(loaders=loaders, *args, **kwargs)
        self.mask_loss = mask_loss
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

    @hx.jit_method(static_argnums=3)
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
        if self.mask_loss:
            mask = batch["legal_moves"]
        else:
            mask = jnp.ones_like(target)
        loss = cross_entropy(output, target, mask)
        loss = jnp.mean(loss, axis=0)
        return loss, output

    @hx.jit_method(static_argnums=3)
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

    @hx.jit_method()
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

            # get accuracy
            accuracy = get_accuracy(output, batch["next_move"])

            # update parameters
            self.params, self.opt_state = self.optimiser_step(grads, self.opt_state, self.params)

            # plot metrics
            self._plot_loss(
                {"loss": loss, "accuracy": accuracy},
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
            loss, output = self.compute_loss(self.params, key, batch, is_training=False)

            # get accuracy
            accuracy = get_accuracy(output, batch["next_move"])

            # track metrics
            losses.append(loss.item())
            self._plot_loss(
                {"loss": loss, "accuracy": accuracy},
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
