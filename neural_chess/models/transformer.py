from typing import Optional

import jax
import numpy as np
import jax.numpy as jnp
import haiku as hk


class MLP(hk.Module):
    def __init__(self, hidden_size, output_size, init_scale):
        """
        Simple 2-layer MLP with GELU activation.
        :param hidden_size:
        :param output_size:
        :param init_scale:
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.init_scale = init_scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        :param x: (..., input_size)
        :return: (..., output_size)
        """
        w_init = hk.initializers.VarianceScaling(self.init_scale)
        x = hk.Linear(self.hidden_size, w_init=w_init)(x)
        x = jax.nn.gelu(x)
        return hk.Linear(self.output_size, w_init=w_init)(x)


class SetTransformer(hk.Module):
    def __init__(self, nb_layers, nb_heads, nb_hidden, nb_embedding, nb_output, dropout=0.1, dropatt=0.1):
        """
        Simple feed-forward transformer stack, with global pooling to a single output.
        :param nb_layers:
        :param nb_heads:
        :param nb_hidden:
        :param nb_embedding:
        :param nb_output:
        :param dropout:
        :param dropatt:
        """
        super().__init__()
        self.nb_layers = nb_layers
        self.nb_heads = nb_heads
        self.nb_hidden = nb_hidden
        self.nb_embedding = nb_embedding
        self.nb_output = nb_output
        self.dropout = dropout
        self.dropatt = dropatt

    def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        """
        :param x: input embeddings (including any conditioning) (..., seq_len, emb_size)
        :return: (..., seq_len, nb_output)
        """
        init_scale = 2.0 / self.nb_layers
        dropout = self.dropout if is_training else 0.0
        dropatt = self.dropatt if is_training else 0.0
