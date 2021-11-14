from pprint import pprint

import pyarrow as pa
from tqdm import tqdm
from typing import List

import chess
import chess.pgn
from chess.pgn import Game
import glob
import multiprocessing as mp

import os

from neural_chess.data_utils.simple_game import SimpleGame

DATA_DIR = "data/"
OUT_FILE = "lichess_300.arrow"
VALID_TIME_CONTROLS = {"300+0"}


class KillSignal:
    # used to signal the worker processes to stop
    pass


# overwrite the default error handling behaviour (i.e don't silently ignore errors)
class CustomGameBuilder(chess.pgn.GameBuilder):
    def handle_error(self, error: Exception) -> None:
        raise IOError("Error parsing game.")


def is_game_valid(game: Game) -> bool:
    """
    Determine whether a game is valid.
    :param game:
    :return:
    """
    return game.headers["TimeControl"] in VALID_TIME_CONTROLS


def read_worker(file_list: List[str], queue: mp.Queue) -> None:
    """
    Read a list of PGN files, and put the resulting `Game` objects into the input queue.
    :param file_list:
    :param queue:
    :return:
    """
    parsed_counter = tqdm(desc="Total games parsed", position=0)
    valid_counter = tqdm(desc="Valid games", position=1)
    for pgn_file in file_list:
        with open(pgn_file, "r") as f:
            while True:
                # attempt to read the next game from the file
                try:
                    game = chess.pgn.read_game(f, Visitor=CustomGameBuilder)
                except IOError as _e:
                    print(f"Error parsing game in {pgn_file}. Skipping.")
                except Exception as e:
                    print(f"Something else went wrong? {e}")

                # terminate at file end
                if game is None:
                    print(f"Reached end of file {pgn_file}.")
                    break

                parsed_counter.update(1)

                # skip invalid games
                if not is_game_valid(game):
                    continue

                # serialise the game and put it on the input queue
                try:
                    game_dict = SimpleGame.from_game(game).serialize()
                except ValueError as e:
                    print(f"Error converting game into `SimpleGame` object.")
                    print(e)
                queue.put(game_dict)
                valid_counter.update(1)
    # kill downstream consumers
    queue.put(KillSignal)
    parsed_counter.close()
    valid_counter.close()
    print("Finished reading files.")


if __name__ == "__main__":
    # grab all pgn files
    pgn_files = glob.glob(os.path.join(DATA_DIR, "*.pgn"))
    print("Loaded PGN files:")
    pprint(pgn_files)

    # create the queue and read process
    q = mp.Queue()
    read_process = mp.Process(target=read_worker, args=(pgn_files, q))
    read_process.start()

    # stream to PyArrow Array File
    nb_kill_signals = 0
    write_buffer = []
    write_freq = 1000
    pa_schema = pa.schema(SimpleGame.pa_fields)
    game_type = pa.struct(SimpleGame.pa_fields)
    out_path = os.path.join(DATA_DIR, OUT_FILE)
    with pa.OSFile(out_path, "wb") as sink, tqdm(desc="Write progress", position=2) as pbar:
        with pa.ipc.new_file(sink, schema=pa_schema) as writer:
            while True:
                item = q.get()
                if item == KillSignal:
                    print("Writing completed.")
                    break
                else:
                    write_buffer.append(item)
                    if len(write_buffer) == write_freq:
                        array = pa.array(write_buffer, type=game_type)
                        batch = pa.RecordBatch.from_struct_array(array)
                        writer.write(batch)
                        write_buffer = []
                pbar.update(1)
    pbar.close()
    read_process.join()
    read_process.close()
