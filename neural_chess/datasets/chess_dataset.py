import chess
import numpy as np
import pyarrow as pa
from chess import Board, Move
from torch.utils.data import Dataset

from neural_chess.utils.data import board_to_flat_repr, get_legal_move_mask, move_to_one_hot, SimpleGame


class ChessDataset(Dataset):
    def __init__(self, path_to_db: str, is_training: bool = True):
        super().__init__()

        # load memory-mapped dataset
        with pa.memory_map(path_to_db, "r") as source:
            self.data: pa.lib.Table = pa.ipc.open_file(source).read_all()

        # 80/20 train/val split
        np.random.seed(42)
        mask = np.random.random(self.data.num_rows) < 0.8
        if is_training:
            (self.indices,) = np.where(mask)

        else:
            (self.indices,) = np.where(~mask)
        print(f"Loaded {len(self.indices)} items for {'train' if is_training else 'test'}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        idx = self.indices[idx]

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

        # whose turn is it? what are their castling rights? what is their ELO?
        turn = board.turn
        castling_rights = board.has_castling_rights(turn)
        if turn == chess.WHITE:
            elo = game.white_elo
        else:
            elo = game.black_elo
        elo = elo / 2500  # approx. in [0, 1]

        # is there an en-passant square?
        # - [0, 63] indicating the position that can be moved to with en-passant
        # - 64 indicating no en-passant rights
        en_passant = board.ep_square if board.ep_square else 64

        legal_moves = get_legal_move_mask(board)

        return {
            "board": flat_repr,
            "next_move": next_move,
            "turn": turn,
            "castling_rights": castling_rights,
            "elo": elo,
            "legal_moves": legal_moves,
            "en_passant": en_passant,
        }

    @staticmethod
    def get_collate_fn():
        def collate_fn(batch):
            """
            Custom collate fn.
            TODO: cast to Jax DeviceArray and transfer to GPU ?
            :param batch:
            :return:
            """
            board = np.stack([item["board"] for item in batch])
            next_move = np.stack([item["next_move"] for item in batch]).astype(np.int32)
            turn = np.array([item["turn"] for item in batch]).astype(np.int32)
            castling_rights = np.array([item["castling_rights"] for item in batch]).astype(np.int32)
            elo = np.array([item["elo"] for item in batch]).astype(np.float32)
            legal_moves = np.stack([item["legal_moves"] for item in batch]).astype(np.bool)
            en_passant = np.array([item["en_passant"] for item in batch]).astype(np.int32)

            return {
                "board_state": board,
                "next_move": next_move,
                "turn": turn,
                "castling_rights": castling_rights,
                "elo": elo,
                "legal_moves": legal_moves,
                "en_passant": en_passant,
            }

        return collate_fn

    @staticmethod
    def get_dummy_batch(batch_size=8):
        """
        Generate a dummy batch of network inputs, for the purpose of initialising the parameter dict.
        Note: this is not a legal board position!
        """
        board = np.random.randint(0, 13, (batch_size, 64)).astype(np.int32)
        turn = np.random.binomial(p=0.5, n=1, size=(batch_size,)).astype(np.int32)
        castling_rights = np.random.binomial(p=0.5, n=1, size=(batch_size,)).astype(np.int32)
        en_passant = np.random.randint(0, 65, (batch_size,)).astype(np.int32)
        elo = np.random.random((batch_size,)).astype(np.float32)
        return {
            "board_state": board,
            "turn": turn,
            "castling_rights": castling_rights,
            "elo": elo,
            "en_passant": en_passant,
        }
