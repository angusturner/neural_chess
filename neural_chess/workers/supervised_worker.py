import numpy as np
from pprint import pprint

import jax
import jax.numpy as jnp
import haiku as hk
from haiku import Transformed

from hijax import AbstractWorker


@jax.jit
def cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the cross entropy loss between logits and labels.
    :param logits: unnormalized log-probabilities
    :param labels: ground-truth labels
    :param mask:
    """
    logits = jnp.where(mask, logits, jnp.full_like(logits, -1e9))
    logits = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(labels * logits, axis=-1)


class SupervisedWorker(AbstractWorker):
    def __init__(self, model, checkpoint_id="best", *args, **kwargs):
        super().__init__(*args, **kwargs)

        # transform the model into a pure jax function
        self.forward = hk.transform(model)
        self.rng = jax.random.PRNGKey(42)
        self.params = None

    def _initialise_parameters(self, loader):
        if self.params is not None:
            return
        # initialise the model
        batch = next(loader.__iter__())
        self.params = self.forward.init(self.rng, is_training=True, **batch)

    def train(self, loader):
        # initialise the network
        self._initialise_parameters(loader)

        @jax.jit
        def compute_loss(params, rng, batch, is_training=True):
            output = self.forward.apply(params, rng, is_training=is_training, **batch)
            target = batch["next_move"]
            mask = batch["legal_moves"]
            loss = cross_entropy(output, target, mask)
            loss = jnp.mean(loss, axis=0)
            return loss, output

        def check_grad(x: jnp.ndarray):
            if jnp.isnan(x).any() or jnp.isinf(x).any():
                raise ValueError("NaN or Inf detected")

        for i, batch in enumerate(loader):
            # forward pass
            (loss, output), grads = jax.value_and_grad(compute_loss, has_aux=True)(self.params, self.rng, batch)

            print("Gradients:")
            jax.tree_multimap(check_grad, grads)
            grad_shapes = jax.tree_multimap(lambda x: x.shape, grads)
            pprint(grad_shapes)

            print(f"Loss: {loss}")

        raise Exception("we made it here!")

    def evaluate(self, loader):
        raise NotImplementedError("")
