import haiku as hk
import jax.numpy as jnp
from typing import Callable, Any

from neural_chess.models.embedding import embed, get_board_pos_emb
from neural_chess.models.transformer import SetTransformer


def build_policy_net(**model_config: Any) -> Callable[..., jnp.ndarray]:
    """
    Build a transformer-based policy network.
    :param model_config:
    :return:
    """
    vocab = model_config["vocab"]
    emb_dim = model_config["embedding_dim"]
    separate_rank_and_file = model_config.get("separate_rank_and_file", False)
    add_result = model_config.get("add_result", False)
    print(f"Adding result? {add_result}")

    def forward(
        board_state: jnp.ndarray,
        turn: jnp.ndarray,
        castling_rights: jnp.ndarray,
        en_passant: jnp.ndarray,
        elo: jnp.ndarray,
        result: jnp.ndarray,
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
        :param result: game result wrt. the player whose turn it is (batch) in [-1, 1]
        :param is_training: whether or not the network is in training mode
        :return:
        """
        # reshape
        batch = board_state.shape[0]
        turn = turn.reshape((batch, 1))
        elo = elo.reshape((batch, 1, 1))
        castling_rights = castling_rights.reshape((batch, 1))
        en_passant = en_passant.reshape((batch, 1))
        result = result.reshape((batch, 1))
        relative_result = result * (turn - 0.5) * 2  # (batch, 1)
        result_feats = jnp.concatenate((relative_result, result), axis=-1).reshape((batch, 1, 2))  # (batch, 1, 2)

        # embed the board state, board positions, etc.
        emb_init = hk.initializers.RandomNormal(stddev=0.02)
        board_state_embedding = embed(board_state, vocab, emb_dim, w_init=emb_init, name="board_emb")  # (B, 64, E)
        board_pos_emb = get_board_pos_emb(emb_dim, separate_rank_and_file)  # (B, 65, E)
        turn_emb = embed(turn, 2, emb_dim, w_init=emb_init, name="turn_emb")  # (B, 1, E)
        castling_rights_emb = embed(castling_rights, 2, emb_dim, w_init=emb_init, name="castle_emb")  # (B, 1, E)
        ep_emb = embed(en_passant, 65, emb_dim, w_init=emb_init, name="en_passant_emb")  # (B, 1, E)

        # project the ELO rating to the embedding dimensions
        w_init = hk.initializers.VarianceScaling(scale=0.1)  # TODO: figure out correct scaling for this
        elo_embedding = hk.Linear(emb_dim, w_init=w_init, name="elo_emb")(elo)  # (B, 1, E)

        # sum all length=1 embeddings, to act in the position of the [cls] token
        cls_token = turn_emb + elo_embedding + castling_rights_emb + ep_emb  # (B, 1, E)

        # concatenate to the front of the board-state embedding, and add the positional embedding
        h = jnp.concatenate([cls_token, board_state_embedding], axis=1)  # (B, 65, E)
        h = h + board_pos_emb

        # replace the final 2 dimension of `h` embedding with the result features
        # note: this is a hack to inject result features, without changing the parameter dictionary.
        if add_result:
            result_feats = jnp.repeat(result_feats, 65, axis=1) * 0.1
            h = jnp.concatenate([h[:, :, :-2], result_feats], axis=-1)  # (B, 65, E)

        return SetTransformer(**model_config)(h, is_training)

    return forward
