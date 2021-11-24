import haiku as hk
import jax
from jax import numpy as jnp


class MLP(hk.Module):
    def __init__(self, hidden_dim, output_dim, init_scale):
        """
        Simple 2-layer MLP with GELU activation.
        :param hidden_dim:
        :param output_dim:
        :param init_scale:
        """
        super().__init__()
        self.hidden_size = hidden_dim
        self.output_size = output_dim
        self.init_scale = init_scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        :param x: (..., input_size)
        :return: (..., output_dim)
        """
        w_init = hk.initializers.VarianceScaling(self.init_scale)
        x = hk.Linear(self.hidden_size, w_init=w_init)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(self.output_size, w_init=w_init)(x)
