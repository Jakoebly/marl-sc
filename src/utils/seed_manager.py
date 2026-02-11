"""Seed management utilities for environment reproducibility."""

from typing import Dict, List, Optional, Tuple
from numpy.random import SeedSequence
import numpy as np


# Define the registry of environment seed names in allocation order
ENVIRONMENT_SEED_REGISTRY: Tuple[str, ...] = (
    'preprocessing',
    'inventory',
    'demand_sampler',
    'lead_time_sampler'
)


class SeedManager:
    """
    Manages seed spawning and conversion for environment components by providin a 
    centralized and extensible way to handle seed allocation.
    """
    
    def __init__(self, root_seed: Optional[int] = None, seed_registry: Optional[Tuple[str, ...]] = None):
        """
        Initializes seed manager with a root seed and a seed registry.
        
        Args:
            root_seed (Optional[int]): Root seed for reproducibility. Defaults to None.
            seed_registry (Optional[Tuple[str, ...]]): Ordered tuple of seed names. 
                If None, uses ENVIRONMENT_SEED_REGISTRY. Defaults to None.
        """

        # Store root seed and seed registry
        self.root_seed = root_seed
        self._original_root_seed = root_seed
        self._episode_counter = 0
        self._seed_registry = seed_registry if seed_registry is not None else ENVIRONMENT_SEED_REGISTRY

        # Initialize seed sequences and spawn seeds
        self._seed_sequences = {}
        self._spawn_seeds()
    
    def _spawn_seeds(self) -> Dict[str, Optional[SeedSequence]]:
        """
        Spawns all seeds from the registry using the root seed.
        
        Returns:
            seed_sequences (Dict[str, Optional[SeedSequence]]): Dictionary mapping seed names to SeedSequence objects.
        """

        # If no root seed is provided, set all seeds to None
        if self.root_seed is None:
            self._seed_sequences = {name: None for name in self._seed_registry}
            return self._seed_sequences.copy()
        
        # Spawn seeds from the root seed
        seed_seq = SeedSequence(self.root_seed)
        spawned = seed_seq.spawn(len(self._seed_registry))
        
        # Store seeds in seed sequences dictionary
        self._seed_sequences = {
            name: seed_seq for name, seed_seq in zip(self._seed_registry, spawned)
        }
        return self._seed_sequences.copy()
    
    def get_seed(self, name: str) -> Optional[SeedSequence]:
        """
        Gets a SeedSequence object for a registered seed name.
        
        Args:
            name (str): Name of the seed to get.
            
        Returns:
            seed_sequence (Optional[SeedSequence]): SeedSequence object or None if not registered.
        """

        # Check if seed name is registered
        if name not in self._seed_registry:
            raise ValueError(f"Seed '{name}' not registered")

        # Get seed sequence from seed sequences dictionary
        seed_seq = self._seed_sequences.get(name)

        return seed_seq
    
    def get_seed_int(self, name: str) -> Optional[int]:
        """
        Gets an integer seed (entropy) for a registered seed name.
        
        Args:
            name (str): Name of the seed to get.
            
        Returns:
            seed_int (Optional[int]): Integer seed or None if not registered.
        """

        # Check if seed name is registered
        if name not in self._seed_registry:
            raise ValueError(f"Seed '{name}' not registered")

        # Get seed sequence from seed sequences dictionary
        seed_seq = self.get_seed(name)

        # Convert seed sequence to integer seed (entropy)
        seed_int = int(seed_seq.generate_state(1, dtype=np.uint32)[0]) if seed_seq is not None else None

        return seed_int
    
    def advance_episode(self) -> None:
        """
        Advances to the next episode's seeds. Call at each env.reset().
        
        If a root seed is set, derives a new deterministic seed from
        (original_root_seed, episode_counter) and re-spawns all component seeds.
        If no root seed is set, this is a no-op (components keep using random RNG).
        """

        # If no original root seed, components stay random (no-op)
        if self._original_root_seed is None:
            return

        # Derive a deterministic per-episode seed from (original_root_seed, episode_counter)
        derived_seed = int(
            SeedSequence([self._original_root_seed, self._episode_counter])
            .generate_state(1, dtype=np.uint32)[0]
        )
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
    
    def get_seeds_for_components(self, component_names: List[str]) -> List[Optional[SeedSequence]]:
        """
        Gets SeedSequence objects for multiple components in order.
        
        Args:
            component_names (List[str]): List of seed names.
            
        Returns:
            seed_seqs (List[Optional[SeedSequence]]): List of SeedSequence objects in the same order.
        """

        # Get seeds for multiple components in order
        seed_seqs = [self.get_seed(name) for name in component_names]

        return seed_seqs
    
    def get_seeds_int_for_components(self, component_names: List[str]) -> List[Optional[int]]:
        """
        Gets integer seeds for multiple components in order.
        
        Args:
            component_names (List[str]): List of seed names.
            
        Returns:
            seed_ints (List[Optional[int]]): List of integer seeds in the same order.
        """

        # Get integer seeds for multiple components in order
        seed_ints = [self.get_seed_int(name) for name in component_names]

        return seed_ints    
    
    @staticmethod
    def seed_to_int(seed: Optional[SeedSequence]) -> Optional[int]:
        """
        Converts a SeedSequence to an integer seed (entropy).
        
        Args:
            seed (Optional[SeedSequence]): SeedSequence to convert.
            
        Returns:
            seed_int (Optional[int]): Integer seed or None if not provided.
        """

        # Convert seed sequence to integer seed (entropy)
        seed_int = int(seed.generate_state(1, dtype=np.uint32)[0]) if seed is not None else None

        return seed_int


def split_seed(root_seed: Optional[int], num_children: int = 2) -> List[Optional[int]]:
    """
    Splits a root seed into multiple independent child seeds using SeedSequence.
    
    Args:
        root_seed (Optional[int]): Root seed to split. If None, returns [None] * num_children.
        num_children (int): Number of child seeds to generate. Defaults to 2.
        
    Returns:
        child_seeds (List[Optional[int]]): List of independent child seeds 
            (or Nones if root_seed is None).
    """

    # If no root seed is provided, return None for each child
    if root_seed is None:
        return [None] * num_children
    
    # Spawn child seeds from the root seed
    children = SeedSequence(root_seed).spawn(num_children)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]