import haiku as hk
import jax.numpy as jnp
from typing import Optional, Callable, Dict, Any

from neural_chess.models.transformer import SetTransformer


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


def build_policy_net(**model_config: Any) -> Callable[..., jnp.ndarray]:
    """
    Build a policy network.
    :param model_config:
    :return:
    """
    vocab = model_config["vocab"]
    emb_dim = model_config["embedding_dim"]

    def forward(
        board_state: jnp.ndarray,
        turn: jnp.ndarray,
        castling_rights: jnp.ndarray,
        en_passant: jnp.ndarray,
        elo: jnp.ndarray,
        is_training: bool = True,
        **_kwargs,
    ) -> jnp.ndarray:
        """
        Forward pass of the policy network.
        :param board_state: state of the board (batch, 64) (int32 ?)
        :param turn: who's turn is it? (batch) in [0, 1]
        :param castling_rights: is the current player allowed to castle? (batch) in [0, 1]
        :param en_passant: which square can we en-passant to (batch) in [0, 64] (64 = no en-passant)
        :param elo: normalised elo rating of the player whose turn it is (batch) (in approx [0, 1])
        :param is_training: whether or not the network is in training mode
        :return:
        """
        # reshape
        batch = board_state.shape[0]
        turn = turn.reshape((batch, -1))
        elo = elo.reshape((batch, 1, 1))
        castling_rights = castling_rights.reshape((batch, 1))
        en_passant = en_passant.reshape((batch, 1))

        # embed the board state, board positions, etc.
        emb_init = hk.initializers.RandomNormal(stddev=0.02)
        board_state_embedding = embed(board_state, vocab, emb_dim, w_init=emb_init, name="board_emb")  # (B, 64, E)
        board_pos = jnp.arange(64).reshape((1, 64))
        board_pos_emb = embed(board_pos, 64, emb_dim, w_init=emb_init, name="board_pos_emb")  # (B, 64, E)
        turn_emb = embed(turn, 2, emb_dim, w_init=emb_init, name="turn_emb")  # (B, 1, E)
        castling_rights_emb = embed(castling_rights, 2, emb_dim, w_init=emb_init, name="castle_emb")  # (B, 1, E)
        ep_emb = embed(en_passant, 65, emb_dim, w_init=emb_init, name="en_passant_emb")  # (B, 1, E)

        # project the ELO rating to the embedding dimensions
        w_init = hk.initializers.VarianceScaling(scale=0.1)  # TODO: figure out correct scaling for this
        elo_embedding = hk.Linear(emb_dim, w_init=w_init, name="elo_emb")(elo)  # (B, 1, E)

        # sum all embeddings into a single hidden state
        h = board_state_embedding + board_pos_emb + turn_emb + elo_embedding + castling_rights_emb + ep_emb

        return SetTransformer(**model_config)(h, is_training)

    return forward
