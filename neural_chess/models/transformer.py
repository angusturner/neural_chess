from typing import Optional

import jax.numpy as jnp
import haiku as hk

from neural_chess.models.mlp import MLP


# helper to create layer-norm on final dimension
def layer_norm(name: Optional[str] = None) -> hk.LayerNorm:
    return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=name)


class SetTransformer(hk.Module):
    def __init__(
        self, nb_layers, nb_heads, hidden_dim, head_dim, output_dim, dropout=0.1, with_pooling=True, **_kwargs
    ):
        """
        Simple feed-forward transformer stack, with no masking or causality.
        Classification layer on the first output with MLP.
        :param nb_layers:
        :param nb_heads:
        :param hidden_dim:
        :param head_dim:
        :param output_dim:
        :param dropout:
        :param with_pooling:
        """
        super().__init__()
        self.nb_layers = nb_layers
        self.nb_heads = nb_heads
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.with_pooling = with_pooling

    def _init_modules_for_layer(self, init_scale):
        """
        Create all the modules for a single layer.
        :param init_scale:
        :return:
        """
        key_size = self.hidden_dim // self.nb_heads
        mha = hk.MultiHeadAttention(num_heads=self.nb_heads, key_size=key_size, w_init_scale=init_scale)
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

        ##
        # [cls] token pooling method
        ln = layer_norm()
        if self.with_pooling:
            h = ln(h)
            h = h[:, 0, :]
            return MLP(hidden_dim=self.hidden_dim * 2, output_dim=self.output_dim, init_scale=init_scale)(h)

        ##
        # retain structure / shape of board, but then re-shape into predictions
        h = MLP(hidden_dim=self.hidden_dim * 2, output_dim=64, init_scale=init_scale)(h)
        h = h[:, 1:, :]  # (batch, 64, 64)
        return h.reshape((-1, self.output_dim))
