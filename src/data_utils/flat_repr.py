import numpy as np
import chess
from chess import Board, Piece
from .piece_mappings import PIECE_TO_INT, INT_TO_PIECE


def board_to_flat_repr(board: Board) -> np.ndarray:
    """
    Convert a `Board` object  to a flattened integer array, as follows:
    1. Represent the board as (8, 8) (rank, file) array
    2. The value of each square takes an integer in [0, 12] (see `PIECE_TO_INT`)
    3. The board is flattened (8, 8) -> (64)
    Note: this representation does not account for the following additional bits of state:
    - Castling rights
    - En passant square
    - Who's turn it is
    :param board:
    :return:
    """
    str_repr = f"{board}"
    out = []
    for line in str_repr.split("\n")[::-1]:
        pieces = line.strip().split(" ")
        for piece in pieces:
            int_repr = PIECE_TO_INT[piece]
            out.append(int_repr)

    return np.array(out)


def flat_repr_to_board(flat_repr: np.ndarray, white_to_move: bool) -> Board:
    """
    Take a flattened integer representation of a board and return a `Board` object
    :param flat_repr:
    :param white_to_move:
    :return:
    """
    board = Board()
    board.turn = white_to_move
    for rank_no, rank in enumerate(flat_repr.reshape(8, 8)):
        for file_no, piece in enumerate(rank):
            symbol = INT_TO_PIECE[piece]
            piece = None if symbol == "." else Piece.from_symbol(symbol)
            square = chess.square(file_index=file_no, rank_index=rank_no)
            board.set_piece_at(square, piece)
    return board
