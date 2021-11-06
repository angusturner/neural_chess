from typing import Optional, List

import chess
import chess.pgn
from chess.pgn import Game
import glob
import multiprocessing as mp

import logging
import os

DATA_DIR = "data/"
VALID_TIME_CONTROLS = {"300+0"}


# overwrite the default error handling behaviour (i.e don't silently ignore errors)
class CustomGameBuilder(chess.pgn.GameBuilder):
    def handle_error(self, error: Exception) -> None:
        raise IOError("Error parsing game.")


def process_pgn_file(pgn_file, break_early: Optional[int] = None) -> List[Game]:
    """
    Read a PGN file, and output a list of `chess.pgn.Game` objects.
    Anything that can be parsed (without errors) is considered valid at this stage.
    :param pgn_file:
    :param break_early:
    :return:
    """
    logging.info("Processing file: {}".format(pgn_file))
    games = []
    with open(pgn_file, "r") as f:
        i = 0
        while True:
            if break_early is not None and i == break_early:
                logging.info(f"Breaking early after {i} games.")
                break
            try:
                game = chess.pgn.read_game(f)
            except IOError as _e:
                logging.error(f"Error parsing game in {pgn_file}. Skipping.")

            if game is None:
                logging.info(f"Reached end of file {pgn_file}")
                break
            games.append(game)

    return games


def is_game_valid(game: Game) -> bool:
    """
    Determine whether a game is valid!
    :param game:
    :return:
    """
    return game.headers["TimeControl"] in VALID_TIME_CONTROLS


if __name__ == "__main__":
    # configure logging
    logging.basicConfig(filename="app.log", filemode="w", format="%(name)s - %(levelname)s - %(message)s")

    # grab all pgn files
    pgn_files = glob.glob(os.path.join(DATA_DIR, "*.pgn"))
    print(pgn_files)
