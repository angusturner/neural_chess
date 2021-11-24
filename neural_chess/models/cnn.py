from typing import Optional

import jax.numpy as jnp
import haiku as hk

from neural_chess.models.mlp import MLP


class CNN_Policy(hk.Module):
    def __init__(self, channels: int = 64, nb_layers: int = 6, kw: int = 3):
        super().__init__()

        self.conv = hk.Conv2D(channels, kernel_shape=kw, padding="same")

        raise NotImplementedError("CNN not implemented.")
