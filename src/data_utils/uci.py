from typing import List

import chess
from chess import Board


def board_to_uci_list(board: Board) -> List[str]:
    """
    Convert a `Board` to a list of moves in UCI notation.
    :param board:
    :return:
    """
    uci_moves = [x.uci() for x in board.move_stack]
    return uci_moves


def uci_list_to_board(uci_moves: List[str]) -> chess.Board:
    """
    Convert a list of moves in UCI notation to a `Board`
    :param uci_moves:
    :return:
    """
    board = chess.Board()
    for move in uci_moves:
        board.push_uci(move)
    return board
