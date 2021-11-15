import numpy as np
from chess import Board

from .one_hot import move_to_one_hot


def get_legal_move_mask(board: Board) -> np.ndarray:
    """
    Return a boolean mask indicating which moves are legal on the next turn.
    :param board:
    :return:
    """
    legal_moves = np.stack([move_to_one_hot(move) for move in board.legal_moves], axis=0)
    legal_mask = np.any(legal_moves, axis=0)
    return legal_mask
