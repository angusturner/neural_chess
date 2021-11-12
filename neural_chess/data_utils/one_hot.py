import chess
import numpy as np
from chess import Square, Move


def square_to_rank_and_file(square: Square) -> (int, int):
    """
    Convert a `chess.Square` into the corresponding rank (1->0) and file (a->0)
    :param square:
    :return:
    """
    return chess.square_rank(square), chess.square_file(square)


def move_to_one_hot(move: chess.Move) -> np.ndarray:
    """
    Convert a `chess.Move` into a one-hot representation, with the following layout:
    (start_rank, start_file, end_rank, end_file) = (8, 8, 8, 8) -> (4096)
    :param move:
    :return:
    """
    out = np.zeros((8, 8, 8, 8), dtype=np.int64)
    start_rank, start_file = square_to_rank_and_file(move.from_square)
    end_rank, end_file = square_to_rank_and_file(move.to_square)
    out[start_rank, start_file, end_rank, end_file] = 1
    return out.flatten()


def one_hot_to_move(one_hot: np.ndarray) -> chess.Move:
    """
    Convert back from one-hot representation to `chess.Move`.
    Note: there is no guarantee that the move is legal!
    """
    x = one_hot.reshape((8, 8, 8, 8))
    ((i, j, k, l),) = zip(*np.where(x))
    start_square = chess.square(rank_index=i, file_index=j)
    end_square = chess.square(rank_index=k, file_index=l)
    return Move(from_square=start_square, to_square=end_square)
