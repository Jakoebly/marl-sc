
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.random import SeedSequence



# Define experiment seed registry
EXPERIMENT_SEEDS: Tuple[str, ...] = (
    'data_weights',
    'data_distances',
    'data_costs',
    'train',
    'eval',
    'obs_stats',
)

# Define environment seed registry
ENVIRONMENT_SEEDS: Tuple[str, ...] = (
    'preprocessing',
    'inventory',
    'demand_sampler',
    'lead_time_sampler',
)

# Define stochastic seed registry
STOCHASTIC_SEEDS: Tuple[str, ...] = (
    'demand_sampler',
    'lead_time_sampler',
)


class SeedManager:
    """
    Manages seed spawning and conversion for environment components by providin a 
    centralized and extensible way to handle seed allocation.
    """

    def __init__(
        self,
        root_seed: Optional[int] = None,
        seed_registry: Tuple[str, ...] = EXPERIMENT_SEEDS,
    ):
        """
        Initializes seed manager with a root seed and a seed registry.

        Args:
            root_seed (Optional[int]): Root seed for reproducibility
            seed_registry (Tuple[str, ...]): Ordered tuple of named seed slots to spawn.
        """

        # Store root seed and seed registries
        self.root_seed = root_seed
        self._original_root_seed = root_seed
        self._episode_counter = 0
        self._seed_registry = seed_registry
        self._seed_sequences: Dict[str, Optional[SeedSequence]] = {}

        # Initialize seed sequences and spawn seeds
        self._spawn_seeds()

    def get_rng(self, name: str) -> np.random.Generator:
        """
        Returns a ``np.random.Generator`` seeded from the named slot. This is the **primary interface**
        and  every consumer of randomness should receive its generator through this method.

        Args:
            name (str): Registered seed name.
        """

        # Get seed sequence from registry
        ss = self._get_seed_sequence(name)

        # Create random number generator from seed sequence
        rng = np.random.default_rng(ss)

        return rng

    def get_seed_int(self, name: str) -> Optional[int]:
        """
        Returns a deterministic integer derived from the named slot.

        Args:
            name (str): Registered seed name.
        """

        # Get seed sequence from registry
        ss = self._get_seed_sequence(name)

        # Return early if seed sequence is None
        if ss is None:
            return None

        # Generate seed integer from seed sequence
        seed_int = int(ss.generate_state(1, dtype=np.uint32)[0])

        return seed_int

    def advance_episode(self) -> None:
        """
        Derives new per-episode seeds from original root seed and episode counter. 
        Called once per episode at each env.reset() to avoid reusing the same seeds.

        """

        # Return early if original root seed is None
        if self._original_root_seed is None:
            return

        # Derive a deterministic per-episode seed from (original_root_seed, episode_counter)
        derived_seed = int(
            SeedSequence([self._original_root_seed, self._episode_counter])
            .generate_state(1, dtype=np.uint32)[0]
        )

        # Update root seed, spawn new seeds, and increment episode counter
        self.root_seed = derived_seed
        self._spawn_seeds()
        self._episode_counter += 1

    def update_root_seed(self, root_seed: Optional[int]) -> None:
        """
        Updates the root seed, resets the episode counter, and respawns all seeds.
        
        Args:
            root_seed (Optional[int]): New root seed.
        """

        # Update root seed, original root seed, and episode counter
        self.root_seed = root_seed
        self._original_root_seed = root_seed
        self._episode_counter = 0

        # Respawn seeds
        self._spawn_seeds()

    @staticmethod
    def derive_env_seed(base_seed: int, worker_index: int, env_index: int) -> int:
        """
        Derives a unique, deterministic seed for a specific parallel environment
        instance from a shared base seed and the worker/environment indices.

        Args:
            base_seed (int): Base seed shared by all environments (e.g., train_seed).
            worker_index (int): EnvRunner worker index (0 for local, 1+ for remote).
            env_index (int): Environment index within the worker.

        Returns:
            seed (int): Unique deterministic seed for this (worker, env) pair.
        """

        # Derive a unique, deterministic seed for a specific parallel environment
        env_seed = int(
            SeedSequence([base_seed, worker_index, env_index])
            .generate_state(1, dtype=np.uint32)[0]
        )
        
        return env_seed

    def _get_seed_sequence(self, name: str) -> Optional[SeedSequence]:
        """
        Retrieves the seed sequence for a given name.
        
        Args:
            name (str): Registered seed name.
        """

        # Check if name is in seed registry
        if name not in self._seed_registry:
            raise ValueError(
                f"Seed '{name}' not in registry {self._seed_registry}"
            )
        
        # Get the seed sequence for the given name
        ss = self._seed_sequences[name]

        return ss

    def _spawn_seeds(self) -> None:
        """
        Spawns new seed sequences for all registered names.
        """

        # Set all seed sequences to None if no root seed is set
        if self.root_seed is None:
            self._seed_sequences = {n: None for n in self._seed_registry}
            return

        # Create a parent seed sequence from the root seed
        parent = SeedSequence(self.root_seed)

        # Spawn child seed sequences from the parent
        children = parent.spawn(len(self._seed_registry))

        # Store the child seed sequences in the seed registry
        self._seed_sequences = dict(zip(self._seed_registry, children))
