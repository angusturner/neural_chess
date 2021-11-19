from typing import Tuple, List

import numpy as np
from chess import Move

from neural_chess.utils.data import one_hot_to_move


def sample_move(move_probs: np.ndarray, greedy: bool = False, topk: int = 5) -> Tuple[Move, List[Tuple[Move, float]]]:
    """
    Given a distribution over moves, sample a
    :param move_probs: the distribution over move probabilities (4096,)
        note: it is assumed that illegal moves have been assigned zero probability!
    :param topk: limit move selection to the topk most likely moves
    :param greedy:
    :return:
    """
    best_move_idxs = np.argsort(-move_probs)[:topk]
    best_move_probs = move_probs[best_move_idxs]
    move_preds = []
    for i, move_idx in enumerate(best_move_idxs):
        # convert to one-hot
        one_hot = np.zeros_like(move_probs)
        one_hot[move_idx] = 1
        move_pred = one_hot_to_move(one_hot)
        move_preds.append((move_pred, best_move_probs[i]))

    # take top move deterministically
    if greedy:
        move = move_preds[0][0]
    # sample multinomial
    else:
        probs = np.array([x[1] for x in move_preds])
        probs = probs / np.sum(probs)  # re-normalise
        outcomes = np.random.multinomial(n=1, pvals=probs)
        (idx,) = np.where(outcomes)[0]
        move = move_preds[idx][0]

    return move, move_preds
