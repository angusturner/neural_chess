from typing import Optional

import haiku as hk
from jax import numpy as jnp


def embed(
    x: jnp.ndarray, vocab: int, emb_dim: int, w_init: hk.initializers.Initializer, name: Optional[str] = None
) -> jnp.ndarray:
    """
    Embed a tensor.
    :param x: tensor to embed
    :param vocab: size of the vocabulary
    :param emb_dim: embedding dimension
    :param name: name of the embedding
    :param w_init: weight initializer
    :return:
    """
    return hk.Embed(vocab, emb_dim, name=name, w_init=w_init)(x)


def get_board_pos_emb(emb_dim: int, separate_rank_and_file: bool = False) -> jnp.ndarray:
    """
    Construct a board embedding for each position on the board, plus one extra position
    which corresponds to the [cls] token.
    :param emb_dim: embedding dimension
    :param separate_rank_and_file:
    :return:
    """
    emb_init = hk.initializers.RandomNormal(stddev=0.02)
    if separate_rank_and_file:
        rank = jnp.arange(8).reshape((1, 8, 1))  # (batch, rank, 1)
        rank_emb = embed(rank, 8, emb_dim, w_init=emb_init, name="rank_embed")  # (batch, rank, 1, emb_dim)
        file = jnp.arange(8).reshape((1, 1, 8))  # (batch, 1, file)
        file_emb = embed(file, 8, emb_dim, w_init=emb_init, name="file_embed")  # (batch, 1, file, emb_dim)
        emb = rank_emb + file_emb  # (batch, rank, file, emb_dim)
        emb = emb.reshape((-1, 64, emb_dim))  # (batch, 64, emb_dim)

        # Pad an extra position in front of the length axis, for the [cls] token.
        emb = jnp.pad(emb, ((0, 0), (1, 0), (0, 0)), mode="constant")  # (B, 65, emb_dim)
    else:
        board_pos = jnp.arange(65).reshape((1, 65))
        emb = embed(board_pos, 65, emb_dim, w_init=emb_init, name="board_pos_emb")  # (B, 64, emb_dim)

    return emb
