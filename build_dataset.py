from pprint import pprint

import io
import pyarrow as pa
from tqdm import tqdm
from typing import List, Optional

import chess
import chess.pgn
from chess.pgn import Game
import glob
import multiprocessing as mp

import os

from neural_chess.utils.data import SimpleGame

DATA_DIR = "data/"
OUT_FILE = "lichess_600_v2.arrow"
VALID_TIME_CONTROLS = {"600+0"}
MIN_HALF_MOVES = 10
MAX_HALF_MOVES = 300
MIN_ELO = 1400
MAX_ELO = 2500
VALID_TERMINATIONS = {"Normal", "Time forfeit"}  # remove 'Abandoned' games, since the result is meaningless
CLOCK_THRESHOLD = 10  # trim moves made with less than 10 seconds left on the clock
BREAK_EARLY: Optional[int] = None  # 1000000  # optionally limit the number of games to process
NB_PROCESSES = 8  # mp.cpu_count() - 1


class KillSignal:
    # used to signal the worker processes to stop
    pass


# overwrite the default error handling behaviour (i.e don't silently ignore errors)
class CustomGameBuilder(chess.pgn.GameBuilder):
    def handle_error(self, error: Exception) -> None:
        raise IOError("Error parsing game.")


def has_valid_game_headers(game: Game) -> bool:
    """
    Determine whether a game is valid based on headers.
    :param game:
    :return:
    """
    if game is None:
        return False
    headers = getattr(game, "headers", None)
    if headers is None:
        return False

    # check that all required headers are present
    required_headers = {"WhiteElo", "BlackElo", "TimeControl", "Termination"}
    for header in required_headers:
        if header not in headers:
            return False

    # check the values of each header
    if headers["TimeControl"] not in VALID_TIME_CONTROLS:
        return False

    # try to parse elo values to integers
    try:
        white_elo = int(headers["WhiteElo"])
        black_elo = int(headers["BlackElo"])
    except ValueError:
        return False

    # check that the elo values are within the valid range
    if white_elo < MIN_ELO or white_elo > MAX_ELO:
        return False
    if black_elo < MIN_ELO or black_elo > MAX_ELO:
        return False

    # only include normal terminations (no abandonment, no time forfeit)
    if headers["Termination"] not in VALID_TERMINATIONS:
        return False

    return True


def is_valid_game(game: SimpleGame) -> bool:
    """
    Determine whether a game is valid, based on the actual move data.
    :param game:
    :return:
    """
    return MIN_HALF_MOVES <= len(game) <= MAX_HALF_MOVES


def read_worker(file_list: List[str], input_q: mp.Queue, nb_consumers: int = 1) -> None:
    """
    Read a list of PGN files, and put the un-parsed string representations into an input queue.
    :param file_list:
    :param input_q:
    :param nb_consumers: number of workers that will consume the input queue
    :return:
    """
    read_counter = tqdm(desc="Total games read.", position=0)
    games_read = 0
    for pgn_file in file_list:
        print("Reading file: {}".format(pgn_file))
        with open(pgn_file, "r") as f:
            n = 0
            game_string = ""
            while True:
                # limit number of lines
                if BREAK_EARLY is not None and games_read >= BREAK_EARLY:
                    print(f"Terminating after {games_read} games.")
                    break

                # grab the next line, until EOF
                line = f.readline()
                if not line:
                    break
                # file structure is <headers>\n<moves>\n<headers>\n<moves>\n...
                # so we accumulate lines into a queue until an even newline is encountered, and
                # then the resulting game is put into the input queue for parsing (and so on)
                if line.strip() == "":
                    n += 1
                    if n > 0 and (n % 2) == 0:
                        input_q.put(game_string)
                        game_string = ""
                        read_counter.update(1)
                        games_read += 1
                    else:
                        game_string += line
                else:
                    game_string += line
    # kill downstream consumers
    for _ in range(nb_consumers):
        input_q.put(KillSignal)
    read_counter.close()
    print("Finished reading files.")


def process_worker(input_q: mp.Queue, output_q: mp.Queue) -> None:
    """
    Consume the input queue, and put the parsed games into the output queue.
    :param input_q:
    :param output_q:
    :return:
    """
    while True:
        game_string = input_q.get()
        if game_string == KillSignal:
            print("Killing process worker.")
            output_q.put(KillSignal)
            break

        # parse the game
        game: Game = chess.pgn.read_game(io.StringIO(game_string), Visitor=CustomGameBuilder)

        # skip games with invalid headers (e.g. missing ELO, invalid time control)
        if not has_valid_game_headers(game):
            continue

        # skip games with invalid move data (e.g. too few moves, too many moves)
        try:
            simple_game = SimpleGame.from_game(game, clock_threshold=CLOCK_THRESHOLD)
            if not is_valid_game(simple_game):
                continue
        except Exception as e:
            print("Error parsing game: {}".format(e))
            continue

        # serialise
        game_dict = simple_game.serialize()
        output_q.put(game_dict)


if __name__ == "__main__":
    # grab all pgn files
    pgn_files = glob.glob(os.path.join(DATA_DIR, "*2021*.pgn"))
    print("Loaded PGN files:")
    pprint(pgn_files)

    # create the input_q and read process
    nb_workers = NB_PROCESSES
    manager = mp.Manager()
    input_queue = manager.Queue(maxsize=10000)
    output_queue = manager.Queue(maxsize=10000)
    read_process = mp.Process(target=read_worker, args=(pgn_files, input_queue, nb_workers))
    read_process.start()

    all_processes = [read_process]
    for i in range(nb_workers):
        process = mp.Process(target=process_worker, args=(input_queue, output_queue))
        process.start()
        all_processes.append(process)

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
                item = output_queue.get()
                if item == KillSignal:
                    nb_kill_signals += 1
                    if nb_kill_signals >= nb_workers:
                        print("Finished writing.")
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

    for process in all_processes:
        process.join()
    print("All processes joined.")
