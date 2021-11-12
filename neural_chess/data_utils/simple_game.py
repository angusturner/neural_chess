from chess.pgn import Game
from chess import Board

from typing import List

from .uci import board_to_uci_list, uci_list_to_board


class SimpleGame:
    # map result-strings to an integer, for classification
    result_map = {"1-0": 0, "0-1": 1, "1/2-1/2": 2}

    def __init__(self, uci_list: List[str], result: int, white_elo: int, black_elo: int):
        """
        Simplified representation of a chess game, containing:
        - A UCI-formatted string of moves
        - The game result (0 = white wins, 1 = black wins, 2 = draw)
        - The ELO rating of each player.
        """
        self.nb_moves = len(uci_list)
        self.moves = uci_list
        self.white_elo = white_elo
        self.black_elo = black_elo
        self.result = result

    @staticmethod
    def from_game(game: Game) -> "SimpleGame":
        """
        Converts a chess.pgn.Game object to a SimpleGame object.
        """
        board = game.end().board()
        moves = board_to_uci_list(board)
        result = SimpleGame.result_map[game.headers["Result"]]
        white_elo = int(game.headers["WhiteElo"])
        black_elo = int(game.headers["BlackElo"])
        return SimpleGame(moves, result, white_elo, black_elo)

    def get_board_at(self, move_index: int) -> Board:
        """
        Returns the board at the given move index
        :param move_index:
        :return:
        """
        assert move_index < self.nb_moves, "Move index must be in [0, nb_moves)"
        board = uci_list_to_board(self.moves[:move_index])
        return board
