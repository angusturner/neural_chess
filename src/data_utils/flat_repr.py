import numpy as np
import chess
from chess import Board, Piece
from typing import Tuple
from .piece_mappings import PIECE_TO_INT, INT_TO_PIECE


def board_to_flat_repr(board: Board) -> Tuple[np.ndarray, bool]:
    """
    Take a `Board` object and represent it as a flattened integer array, indicating
    the piece value of each square. Additionally return a boolean indicating if
    white is moving next.
    Note: row 0 of the board is rank 8, and col 0 is file a
    TODO: Consider the following additional pieces of info, which also determine the state
        - is en-passant available on the next turn?
        - is castling still permitted?
        - what is the move count?
    :param board:
    :return:
    """
    str_repr = f"{board}"
    out = []
    for line in str_repr.split("\n"):
        pieces = line.strip().split(" ")
        for piece in pieces:
            int_repr = PIECE_TO_INT[piece]
            out.append(int_repr)

    return np.array(out), board.turn


def flat_repr_to_board(flat_repr: np.ndarray, white_to_move: bool) -> Board:
    """
    Take a flattened integer representation of a board and return a `Board` object
    :param flat_repr:
    :param white_to_move:
    :return:
    """
    board = Board()
    board.turn = white_to_move
    rank_no = 7
    for rank in flat_repr.reshape(8, 8):
        file_no = 0
        for piece in rank:
            symbol = INT_TO_PIECE[piece]
            piece = None if symbol == "." else Piece.from_symbol(symbol)
            square = chess.square(file_index=file_no, rank_index=rank_no)
            board.set_piece_at(square, piece)
            file_no += 1
        rank_no -= 1
    return board
