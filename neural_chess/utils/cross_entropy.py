import jax
from jax import numpy as jnp


@jax.jit
def cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the cross entropy loss between logits and labels.
    :param logits: unnormalized log-probabilities
    :param labels: ground-truth labels
    :param mask: mask the probabilities before normalising them, to prevent the model
        assigning weight to illegal moves
    """
    logits = jnp.where(mask, logits, jnp.full_like(logits, -1e9))
    logits = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(labels * logits, axis=-1)
