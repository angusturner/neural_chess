import chess
import numpy as np
import pyarrow as pa
from chess import Board, Move
from torch.utils.data import Dataset

from neural_chess.data_utils import board_to_flat_repr
from neural_chess.data_utils.one_hot import move_to_one_hot
from neural_chess.data_utils.simple_game import SimpleGame


class ChessDataset(Dataset):
    def __init__(self, path_to_db: str):
        super().__init__()

        # load memory-mapped dataset
        with pa.memory_map(path_to_db, "r") as source:
            self.data: pa.lib.Table = pa.ipc.open_file(source).read_all()

    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """

        # get the game
        keys = ("moves", "white_elo", "black_elo", "result")
        game_dict = {}
        for key in keys:
            game_dict[key] = self.data[key][idx].as_py()
        game = SimpleGame.from_serialized(game_dict)

        # sample a random move, excluding the final move
        move_idx = int(np.random.randint(0, len(game) - 1))
        next_move = game.moves[move_idx + 1]

        # get the board encoding at this position
        board: Board = game.get_board_at(move_idx)
        flat_repr = board_to_flat_repr(board).astype(np.int32)

        # encode the next move into a one-hot target
        next_move = move_to_one_hot(Move.from_uci(next_move))

        # get the player's turn and castling rights
        turn = board.turn
        castling_rights = board.has_castling_rights(turn)

        # get the players ELO score
        if turn == chess.WHITE:
            elo = game.white_elo
        else:
            elo = game.black_elo

        return {
            "board": flat_repr,
            "next_move": next_move,
            "turn": turn,
            "castling_rights": castling_rights,
            "elo": elo,
        }

    def get_collate_fn(self):
        def collate_fn(batch):
            pass

        raise NotImplementedError("Oops")
