"""
Represent each piece as a symbol, with `.` for the empty square
"""
black_pieces = ["k", "q", "b", "n", "r", "p"]
white_pieces = [x.upper() for x in black_pieces]
all_pieces = white_pieces + black_pieces + ["."]

INT_TO_PIECE = {i: v for i, v in enumerate(all_pieces)}
PIECE_TO_INT = {v: k for k, v in INT_TO_PIECE.items()}
