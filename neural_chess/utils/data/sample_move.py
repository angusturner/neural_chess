from typing import Tuple, List

import numpy as np
from chess import Move

from neural_chess.utils.data import one_hot_to_move


def softmax(x, temp: float = 1.0):
    """
    Compute softmax with a temperature parameter.
    :param x:
    :param temp:
    :return:
    """
    x = x / temp
    x = np.exp(x - np.max(x))
    x = x / np.sum(x)
    return x


def sample_multinomial(p):
    """
    Sample multinomial distribution.
    """
    x = np.random.uniform(0, 1)
    for i, v in enumerate(np.cumsum(p)):
        if x < v:
            return i
    return len(p) - 1  # shouldn't happen...


def sample_move(
    move_probs: np.ndarray, greedy: bool = False, topk: int = 5, temp: float = 1.0
) -> Tuple[Move, List[Tuple[Move, float]]]:
    """
    Given a distribution over moves, sample a
    :param move_probs: the distribution over move probabilities (4096,)
        note: it is assumed that illegal moves have been assigned zero probability!
    :param topk: limit move selection to the topk most likely moves
    :param temp: temperature for sampling
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
        log_probs = np.log(probs + 1e-8)
        probs = softmax(log_probs, temp=temp)
        idx = sample_multinomial(probs)
        print(f"Sampling move: {idx}")
        move = move_preds[idx][0]

    return move, move_preds
