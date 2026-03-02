from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class StochasticComponent(ABC):
    """
    Implements a base class for components that use randomness. All stochastic 
    components should inherit from this class and implement a reset() method to 
    to accept a ``np.random.Generator`` provided by the ``SeedManager``.
    """

    @abstractmethod
    def reset(self, rng: Optional[np.random.Generator] = None):
        """
        Resets the component's random state.

        Args:
            rng (Optional[np.random.Generator]): Generator provided by the SeedManager. If ``None``,
                the component creates an unseeded generator.
        """
        pass
