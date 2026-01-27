"""Seed management utilities for environment reproducibility."""

from typing import Dict, List, Optional, Tuple
from numpy.random import SeedSequence


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
        seed_int = seed_seq.entropy if seed_seq is not None else None

        return seed_int
    
    def update_root_seed(self, root_seed: Optional[int]) -> None:
        """
        Updates the root seed and respawns all seeds.
        
        Args:
            root_seed (Optional[int]): New root seed.
        """

        # Update root seed and respawn seeds
        self.root_seed = root_seed
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
        seed_int = seed.entropy if seed is not None else None

        return seed_int

