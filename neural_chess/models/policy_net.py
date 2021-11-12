import haiku as hk
import jax.numpy as jnp
from typing import Optional, Callable

from neural_chess.models.transformer import SetTransformer


def build_policy_net(model_config: Optional[dict] = None) -> Callable[..., jnp.ndarray]:
    if model_config is None:
        model_config = {}

    vocab = model_config["vocab"]
    embedding_dim = model_config["embedding_dim"]

    def forward(board_state: jnp.ndarray, turn: jnp.ndarray, elo: jnp.ndarray, is_training: bool = True) -> jnp.ndarray:
        """
        Forward pass of the policy network.
        :param board_state: state of the board (batch, 64) (int32 ?)
        :param turn: who's turn is it? (batch)
        :param elo: normalised elo rating of the player whose turn it is (batch) (in approx [0, 1])
        :param is_training: whether or not the network is in training mode
        :return:
        """
        # reshape
        batch = board_state.shape[0]
        turn = turn.reshape((batch, -1))
        elo = elo.reshape((batch, 1, 1))

        # embed the board state, board positions, and turn
        embed_init = hk.initializers.RandomNormal(stddev=0.1)
        board_state_embedding = hk.Embed(vocab, embedding_dim, w_init=embed_init)(board_state)
        board_positions = jnp.arange(64).reshape((1, 64))
        board_positions_embedding = hk.Embed(64, embedding_dim, w_init=embed_init)(board_positions)
        turn_embedding = hk.Embed(2, embedding_dim, w_init=embed_init)(turn)

        # project the ELO rating to the embedding dimensions
        w_init = hk.initializers.VarianceScaling(scale=0.1)  # ? not sure exactly how to set this
        elo_embedding = hk.Linear(embedding_dim, w_init=w_init)(elo)

        # sum all embeddings into a single hidden state
        # (batch, 64, embedding_dim), (batch, 64, embedding_dim), (batch, 1, embedding_dim), (batch, 1, embedding_dim)
        h = board_state_embedding + board_positions_embedding + turn_embedding + elo_embedding

        return SetTransformer(**model_config)(h, is_training)

    return forward
