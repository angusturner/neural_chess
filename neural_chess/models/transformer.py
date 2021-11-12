from typing import Optional

import jax
import numpy as np
import jax.numpy as jnp
import haiku as hk


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


# helper to create layer-norm on final dimension
def layer_norm(name: Optional[str] = None) -> hk.LayerNorm:
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)


class SetTransformer(hk.Module):
    def __init__(self, nb_layers, nb_heads, hidden_dim, head_dim, output_dim, dropout=0.1):
        """
        Simple feed-forward transformer stack, with no masking or causality.
        Global pooling to a single output.
        :param nb_layers:
        :param nb_heads:
        :param hidden_dim:
        :param head_dim:
        :param output_dim:
        :param dropout:
        """
        super().__init__()
        self.nb_layers = nb_layers
        self.nb_heads = nb_heads
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.output_dim = output_dim
        self.dropout = dropout

    def _init_modules_for_layer(self, init_scale):
        """
        Create all the modules for a single layer.
        :param init_scale:
        :return:
        """
        mha = hk.MultiHeadAttention(num_heads=self.nb_heads, key_size=self.hidden_dim, w_init_scale=init_scale)
        mlp = MLP(hidden_dim=self.hidden_dim * 2, output_dim=self.hidden_dim, init_scale=init_scale)
        ln1 = layer_norm()
        ln2 = layer_norm()
        return mha, mlp, ln1, ln2

    def __call__(self, x: jnp.ndarray, is_training: bool) -> jnp.ndarray:
        """
        :param x: input embeddings (including any conditioning) (..., seq_len, emb_size)
        :return: (..., seq_len, nb_output)
        """
        init_scale = 2.0 / self.nb_layers
        dropout = self.dropout if is_training else 0.0

        h = x
        for i in range(self.nb_layers):
            mha, mlp, ln1, ln2 = self._init_modules_for_layer(init_scale)
            h_norm = ln1(h)
            h_attn = mha(h_norm, h_norm, h_norm)
            h_attn = hk.dropout(hk.next_rng_key(), dropout, h_attn)
            h = h_attn + h
            h_norm = ln2(h)
            h_dense = mlp(h_norm)
            h_dense = hk.dropout(hk.next_rng_key(), dropout, h_dense)
            h = h + h_dense

        h = layer_norm(h)

        # average over length dimension
        h = jnp.mean(h, axis=-2)

        # project to final output
        return hk.Linear(self.output_dim, w_init=hk.initializers.VarianceScaling(init_scale))(h)
