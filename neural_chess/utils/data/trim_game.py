from chess import Board
from chess.pgn import Game


def trim_game(game: Game, min_clock: float = 15) -> Board:
    """
    Convert a game to a board, trimming out moves made under time pressure.
    :param game:
    :param min_clock:
    :return:
    """
    node = game
    while True:
        next_node = node.next()
        if next_node is None:
            break
        if next_node.clock() < min_clock:
            break
        node = next_node

    return node.board()
