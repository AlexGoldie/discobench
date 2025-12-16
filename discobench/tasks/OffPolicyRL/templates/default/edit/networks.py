import flax.linen as nn
import jax.numpy as jnp
from typing import Callable

class QNetwork(nn.Module):
    action_dim: int
    width: int
    depth: int
    activation: Callable

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        """Insert your network logic here."""
        # Input = x. x is the environment
        # Shape = (batch_size, obs_dim)

        # Calculate Q-values for each action
        q = ...

        return q # shape (batch_size, action_dim)
