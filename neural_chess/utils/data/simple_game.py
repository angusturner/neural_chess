import pyarrow as pa
from chess.pgn import Game
from chess import Board

from typing import List, Optional

from .trim_game import trim_game
from .uci import board_to_uci_list, uci_list_to_board


class SimpleGame:
    """
    Simplified representation of a chess game, containing the move list,
    player rankings and the result. Contains information on how to serialise this representation into a PyArrow struct,
    for efficient storage and memory-mapped retrieval.
    """

    # map result-strings to an integer
    result_map = {"1-0": 0, "0-1": 1, "1/2-1/2": 2}

    # define the serialization format for PyArrow
    pa_fields = [
        ("moves", pa.list_(pa.string())),
        ("white_elo", pa.uint16()),
        ("black_elo", pa.uint16()),
        ("result", pa.uint8()),
    ]
    pa_type = pa.struct(pa_fields)

    def __init__(self, uci_list: List[str], result: int, white_elo: int, black_elo: int):
        """
        :param uci_list: UCI-formatted string of moves
        :param result: The game result (0 = white wins, 1 = black wins, 2 = draw)
        :param white_elo: The ELO rating of the white player
        :param black_elo: The ELO rating of the black player
        """
        self.moves = uci_list
        self.white_elo = white_elo
        self.black_elo = black_elo
        self.result = result

    def __len__(self):
        return len(self.moves)

    @staticmethod
    def from_game(game: Game, clock_threshold: Optional[int] = None) -> "SimpleGame":
        """
        Converts a chess.pgn.Game object to a SimpleGame object.
        :param game: The chess.pgn.Game object to convert
        :param clock_threshold: If set, moves made under time pressure will be truncated
        """
        if clock_threshold is None:
            board = game.end().board()
        else:
            board = trim_game(game, clock_threshold)
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
        assert move_index < len(self), "Move index must be in [0, nb_moves)"
        board = uci_list_to_board(self.moves[: move_index + 1])
        return board

    def serialize(self) -> dict:
        return {"moves": self.moves, "result": self.result, "white_elo": self.white_elo, "black_elo": self.black_elo}

    @staticmethod
    def from_serialized(serialized: dict) -> "SimpleGame":
        return SimpleGame(
            uci_list=serialized["moves"],
            result=serialized["result"],
            white_elo=serialized["white_elo"],
            black_elo=serialized["black_elo"],
        )
