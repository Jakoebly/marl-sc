import numpy as np
from gymnasium.spaces import Space, Box, Discrete, MultiDiscrete, Dict as GymDict, Tuple as GymTuple
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override

from src.algorithms.models.registry import get_architecture


class BaseRLModule(TorchRLModule):
    """
    Implements a general-purpose base class for all RLModules by
    providing common utilities and GRU handling.
    """

    @staticmethod
    def build_network(architecture_type: str, input_dim: int, output_dim: int, architecture_config: Dict[str, Any], name: str = "") -> nn.Module:
        """
        Builds a network module from architecture config.
        
        Args:
            architecture_type (str): Architecture name (e.g., "mlp", "gru", "cnn")
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            architecture_config (Dict[str, Any]): Architecture-specific config dict
            name (str): Optional name prefix
            
        Returns:
            nn.Module: PyTorch module (Sequential for MLP/CNN, ModuleDict for GRU)
        """

        # Get the architecture class and instance
        arch_class = get_architecture(architecture_type)
        arch_instance = arch_class()

        # Build the network
        network = arch_instance.build(input_dim, output_dim, architecture_config, name)

        return network

    def _is_gru_from_network(self, network: nn.Module) -> bool:
        """
        Check if network is a GRU architecture based on whether it is a ModuleDict with a "gru" key.
        
        Args:
            network (nn.Module): Network module to check.
            
        Returns:
            bool: True if network is GRU (ModuleDict with "gru" key), False otherwise.
        """

        return isinstance(network, nn.ModuleDict) and "gru" in network

    def _forward_network(self, network: nn.Module, inputs: torch.Tensor, hidden_states:  Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Performs a forward pass through a given network module.
        
        Args:
            network (nn.Module): Network module to forward through.
            inputs (torch.Tensor): Input tensor.
                Shape: (B, obs_dim) or (B, seq_len, obs_dim)
            hidden_states (Optional[torch.Tensor]): Hidden states inputs for RNN layers.
                Shape: (B, num_layers*num_directions, hidden_size)
            
        Returns:
            network_out (torch.Tensor): Network output.
                Shape: (B, output_dim) or (B, seq_len, output_dim)
            network_states_out (Optional[torch.Tensor]): Hidden states output for all RNN layers.
                Shape: (B, num_layers*num_directions, hidden_size) or None
        """

        # If network is a GRU, forward using the _forward_gru() method
        if self._is_gru_from_network(network):
            network_out, network_states_out = self._forward_gru(network, inputs, hidden_states) # Shape: (B, output_dim) or (B, seq_len, output_dim), (B, num_layers*num_directions, hidden_size)

        # If network is not a GRU, forward through the network module directly
        else:
            network_out = network(inputs) # Shape: (B, output_dim) or (B, seq_len, output_dim)
            network_states_out = None # Shape: None

        return network_out, network_states_out

    def _forward_gru(self, network: nn.ModuleDict, inputs: torch.Tensor, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through a given GRU network module.
        
        Args:
            network (nn.ModuleDict): GRU network module containing a "gru" and "output_proj" submodule.
            inputs (torch.Tensor): Input tensor.
                Shape: (B, obs_dim) or (B, seq_len, obs_dim)
            hidden_states (Optional[torch.Tensor]): Hidden states inputs for RNN layers.
                Shape: (B, num_layers*num_directions, hidden_size)
            
        Returns:
            network_out (torch.Tensor): Network output.
                Shape: (B, output_dim) or (B, seq_len, output_dim)
            network_states_out (Optional[torch.Tensor]): Hidden states output for all RNN layers.
                Shape: (B, num_layers*num_directions, hidden_size)
        """

        # Extract the GRU and output projection submodules
        gru = network["gru"]
        output_proj = network["output_proj"]

        # Add sequence dimension to inputs if not present (nn.GRU expects (B, seq_len, input_dim) with batch_first=True)
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(1)  # Shape: (B, input_dim) -> (B, 1, input_dim)
        
        # Add batch dimension to hidden states if not there
        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states.unsqueeze(0) # Shape: (num_layers*num_directions, hidden_size) -> (1, num_layers*num_directions, hidden_size)
    
        # Swap batch and layer dimensions (nn.GRU expects (num_layers*num_directions, B, hidden_size))
        hidden_states = hidden_states.transpose(0, 1) # Shape: (B, num_layers*num_directions, hidden_size) -> (num_layers*num_directions, B, hidden_size)

        # Forward through GRU
        gru_out, gru_states_out = gru(inputs, hidden_states) # Shape: (B, seq_len, hidden_size*num_directions), (num_layers*num_directions, B, hidden_size)
        
        # Swap batch and layer dimensions again to match RLlib's expected output shape
        gru_states_out = gru_states_out.transpose(0, 1) # Shape: (B, num_layers*num_directions, hidden_size)

        # Forward through output projection
        network_out = output_proj(gru_out) # Shape: (B, seq_len, output_dim)

        return network_out, gru_states_out


class ActorCriticRLModule(BaseRLModule, ValueFunctionAPI):
    """
    Implements a base class for actor-critic algorithms by
    initializing and forwarding actor and critic networks.
    """

    def setup(self):
        """
        Initializes the actor-critic RLModule by building the shared, actor, and critic networks.
        """

        # Get the observation and action spaces
        obs_space = self.observation_space
        local_obs_dim = self._get_obs_dim(obs_space, key="local")
        global_obs_dim = self._get_obs_dim(obs_space, key="global")
        action_space = self.action_space
        action_dim = self._get_action_dim(action_space)
        actor_output_dim = self._get_actor_output_dim(action_space)

        # Extract network configurations from model config
        network_configs = self.model_config.get("networks", {})

        # Extract shared layers configuration
        shared_layers_dict = network_configs.get("shared_layers", {}) or {}
        self.shared_layers_config = shared_layers_dict.get("config", {})
        self.shared_layers_type = shared_layers_dict.get("type", "")

        # Extract actor configuration
        actor_dict = network_configs.get("actor") or {}
        self.actor_config = actor_dict.get("config", {})
        self.actor_type = actor_dict.get("type", "")

        # Extract critic configuration
        critic_dict = network_configs.get("critic") or {}
        self.critic_config = critic_dict.get("config", {})
        self.critic_type = critic_dict.get("type", "")

        # Set flags
        self.has_shared_layers = network_configs.get("shared_layers") is not None
        self.use_centralized_critic = self.model_config.get("use_centralized_critic", False)
        
        # Build shared layers if specified
        if self.has_shared_layers:
            shared_layers_output_dim = self.shared_layers_config.get("output_dim", 128)
            self.shared_layers = self.build_network(
                architecture_type=self.shared_layers_type,
                input_dim=local_obs_dim,
                output_dim=shared_layers_output_dim,
                architecture_config=self.shared_layers_config,
                name="shared"
            )
            actor_input_dim = shared_layers_output_dim
            critic_input_dim = global_obs_dim if self.use_centralized_critic else shared_layers_output_dim
        else:
            actor_input_dim = local_obs_dim
            critic_input_dim = global_obs_dim if self.use_centralized_critic else local_obs_dim
        
        # Build actor network
        self.actor = self.build_network(
            architecture_type=self.actor_type,
            input_dim=actor_input_dim,
            output_dim=actor_output_dim,
            architecture_config=self.actor_config,
            name="actor"
        )

        # Build critic network with local obs as default input dimension
        self.critic = self.build_network(
            architecture_type=self.critic_type,
            input_dim=critic_input_dim, 
            output_dim=1,  
            architecture_config=self.critic_config,
            name="critic"
        )

    def _get_obs_dim(self, space: Space, key: Optional[str] = None) -> int:
        """
        Gets dimensions of a given observation space.

        Args:
            space (Space): Observation space to get dimensions of.
            key (Optional[str]): Key to get dimensions of if space is a GymDict. If None, returns dim(space).

        Returns:
            int: Dimensions of the observation space.

        """

        # If space is a Box, ignore `key` and return the dimension of the space 
        if isinstance(space, Box):
            return int(np.prod(space.shape))
        if isinstance(space, Discrete):
            return space.n
        if isinstance(space, MultiDiscrete):
            return int(np.sum(space.nvec))
        if isinstance(space, GymTuple):
            return sum(self._get_obs_dim(s) for s in space.spaces)
        if isinstance(space, GymDict):
            if key is None:
                return sum(self._get_obs_dim(s) for s in space.spaces.values())
            if key not in space.spaces:
                raise KeyError(
                    f"Dict obs space keys={list(space.spaces.keys())}, "
                    f"requested key='{key}'"
                )
            return self._get_obs_dim(space.spaces[key])
        raise NotImplementedError(f"Unsupported observation space: {space}")
    
    def _get_action_dim(self, action_space: Space) -> int:
        """
        Gets dimensions of a given action space.
        
        Args:
            action_space (Space): Action space to get dimensions of.
            
        Returns:
            int: Dimensions of the action space.
        """

        # Get dimensions of the action space
        if isinstance(action_space, Box):
            return int(np.prod(action_space.shape))
        if isinstance(action_space, Discrete):
            return action_space.n
        if isinstance(action_space, MultiDiscrete):
            return int(np.sum(action_space.nvec))
        raise NotImplementedError(f"Unsupported action space: {action_space}")
    
    def _get_actor_output_dim(self, action_space: Space) -> int:
        """
        Gets dimensions of the output of the actor network that are expected by RLlib.
        
        Args:
            action_space (Space): Action space for the actor network.
                
        Returns:
            int: Dimensions of the output of the actor network.
        """

        # Get dimensions of the output of the actor network
        if isinstance(action_space, Box):
            return 2 * self._get_action_dim(action_space)  # Mean + log_std
        elif isinstance(action_space, Discrete):
            return self._get_action_dim(action_space)  # Logits
        elif isinstance(action_space, MultiDiscrete):
            # For MultiDiscrete, sum of all action dimensions (logits for each discrete space)
            return self._get_action_dim(action_space)
        else:
            raise NotImplementedError(f"Unsupported action space type: {type(action_space)}")
        
    def get_initial_state(self) -> Dict[str, Any]:
        """
        Gets initial hidden states for GRU networks.
        
        Returns:
            states (Dict[str, Any]): Dictionary of initial hidden states.
                Shape: Dict of {layer_name: (num_layers*num_directions, hidden_size)}
        """

        # Initialize dictionary of initial hidden states
        states = {}

        # Get initial hidden states for shared layers if specified and if it is a GRU
        if self.has_shared_layers:
            if self.shared_layers_type == "gru":
                num_layers = self.shared_layers_config.get("num_layers", 2)	
                hidden_size = self.shared_layers_config.get("hidden_size", 128)
                bidirectional = self.shared_layers_config.get("bidirectional", False)
                num_directions = 2 if bidirectional else 1
                states["shared_h"] = torch.zeros(num_layers * num_directions, hidden_size)

        # Get initial hidden states for actor network if it is a GRU
        if self.actor_type == "gru":
            num_layers = self.actor_config.get("num_layers", 2)	
            hidden_size = self.actor_config.get("hidden_size", 128)
            bidirectional = self.actor_config.get("bidirectional", False)
            num_directions = 2 if bidirectional else 1
            states["actor_h"] = torch.zeros(num_layers * num_directions, hidden_size)

        # Get initial hidden states for critic network
        if self.critic_type == "gru":
            num_layers = self.critic_config.get("num_layers", 2)	
            hidden_size = self.critic_config.get("hidden_size", 128)
            bidirectional = self.critic_config.get("bidirectional", False)
            num_directions = 2 if bidirectional else 1
            states["critic_h"] = torch.zeros(num_layers * num_directions, hidden_size)
        
        return states

    def _forward_shared(self, obs: torch.Tensor, hidden_states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Performs a forward pass through shared layers.
        
        Args:
            obs (torch.Tensor): Input observations.
                Shape: (B, obs_dim) or (B, seq_len, obs_dim)
            hidden_states (Optional[torch.Tensor]): Hidden states inputs for all RNN layers.
                Shape: (B, num_layers*num_directions, hidden_size)

        Returns:
            shared_out (torch.Tensor): Shared layers network output.
                Shape: (B, shared_dim) or (B, seq_len, shared_dim)
            shared_states_out (Optional[torch.Tensor]): Hidden states output for all RNN layers.
                Shape: (B, num_layers*num_directions, hidden_size)
        """

        # Forward through shared layers using the base class method
        shared_out, shared_states_out = self._forward_network(self.shared_layers, obs, hidden_states)

        return shared_out, shared_states_out

    def _forward_actor(self, obs: torch.Tensor, hidden_states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Performs a forward pass through the actor network.
        
        Args:
            obs (torch.Tensor): Input observations.
                Shape: (B, obs_dim) or (B, seq_len, obs_dim)
            hidden_states (Optional[torch.Tensor]): Hidden states inputs for all RNN layers.
                Shape: (B, num_layers*num_directions, hidden_size)

        Returns:
            actor_out (torch.Tensor): Actor network output.
                Shape: (B, action_dim) or (B, seq_len, action_dim)
            actor_states_out (Optional[torch.Tensor]): Hidden states output for all RNN layers.
                Shape: (B, num_layers*num_directions, hidden_size)
        """

        # Forward through actor network using the base class method
        actor_out, actor_states_out = self._forward_network(self.actor, obs, hidden_states)

        return actor_out, actor_states_out

    def _forward_critic(self, obs: torch.Tensor, hidden_states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Performs a forward pass through the critic network.
        
        Args:
            obs (torch.Tensor): Input observations.
                Shape: (B, obs_dim) or (B, seq_len, obs_dim)
            hidden_states (Optional[torch.Tensor]): Hidden states inputs for all RNN layers.
                Shape: (B, num_layers*num_directions, hidden_size)
        
        Returns:
            critic_out (torch.Tensor): Critic network output.
                Shape: (B, 1) or (B, seq_len, 1)
            critic_states_out (Optional[torch.Tensor]): Hidden states output for all RNN layers.
                Shape: (B, num_layers*num_directions, hidden_size)
        """

        # Forward through critic network using the base class method
        critic_out, critic_states_out = self._forward_network(self.critic, obs, hidden_states)

        return critic_out, critic_states_out

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Performs a forward pass during action sampling (inference). This method is called by 
        RLlib to get the output of the network to (deterministically) determine the next 
        actions. It returns action distribution inputs (logits/parameters), which RLlib will 
        use to create a deterministic distribution (e.g., argmax, ...) to sample actions from.
        
        Args:
            batch (Dict[str, Any]): Batch dictionary containing at least the following columns:
                - Columns.OBS (torch.Tensor): Input observations.
                    Shape: (B, obs_dim) or (V, seq_len, obs_dim)
                - Columns.STATE_IN (Dict[str, torch.Tensor]): Hidden states inputs for all RNN layers.
                    Shape: Dict of {layer_name: (B, num_layers*num_directions, hidden_size)}
        
        Returns:
            result (Dict[str, Any]): Dictionary containing the output of the network:
                - Columns.ACTION_DIST_INPUTS (torch.Tensor): Action distribution inputs.
                    Shape: (B, action_dim) or (B, seq_len, action_dim)
                - Columns.STATE_OUT (Dict[str, torch.Tensor]): Hidden states output for all RNN layers.
                    Shape: Dict or single of (num_layers*num_directions, B, hidden_size)
        """

        # Extract observation
        obs = batch.get(Columns.OBS) # Shape: (B, obs_dim) or (B, seq_len, obs_dim)
        if obs is None:
            raise ValueError("Missing Columns.OBS in batch")
        local_obs = obs.get("local")
        global_obs = obs.get("global")
        if local_obs is None or global_obs is None:
            raise ValueError("Expected 'local' and 'global' keys in observation dict")

        # Extract hidden states from batch 
        states_in = batch.get(Columns.STATE_IN, {}) # Shape: {} or {layer_name: (B, num_layers*num_directions, hidden_size)}

        # Forward through shared layers if existing
        shared_states_out = None
        if self.has_shared_layers:
            shared_hidden = states_in.get("shared_h") # Shape: None or {layer_name: (B, num_layers*num_directions, hidden_size)}
            shared_out, shared_states_out = self._forward_shared(local_obs, shared_hidden) # Shape: (B, shared_dim) or (B, seq_len, shared_dim), None or {layer_name: (B, num_layers*num_directions, hidden_size)}
            local_obs = shared_out

        # Forward through actor network
        actor_hidden = states_in.get("actor_h") # Shape: None or {layer_name: (B, num_layers*num_directions, hidden_size)}
        actor_out, actor_states_out = self._forward_actor(local_obs, actor_hidden) # Shape: (B, action_dim) or (B, seq_len, action_dim), None or {layer_name: (B, num_layers*num_directions, hidden_size)}

        # Forward through critic network if it is a GRU to obtain hidden states for next step
        critic_states_out = None
        if self._is_gru_from_network(self.critic):
            critic_hidden = states_in.get("critic_h") # Shape: None or {layer_name: (B, num_layers*num_directions, hidden_size)}
            critic_obs = global_obs if self.use_centralized_critic else local_obs # Shape: (B, obs_dim) or (B, seq_len, obs_dim)
            _, critic_states_out = self._forward_critic(critic_obs, critic_hidden) # Shape: (B, 1) or (B, seq_len, 1), None or {layer_name: (B, num_layers*num_directions, hidden_size)}

        # Create final states_out dictionary
        # Pass through all keys from states_in unchanged
        states_out = dict(states_in)
        # Overwrite with processed states if they exist
        if shared_states_out is not None:
            states_out["shared_h"] = shared_states_out
        if actor_states_out is not None:  
            states_out["actor_h"] = actor_states_out
        if critic_states_out is not None:
            states_out["critic_h"] = critic_states_out

        # Create final result dictionary
        result = {
            Columns.ACTION_DIST_INPUTS: actor_out,
        }
        if states_out:
            result[Columns.STATE_OUT] = states_out

        return result

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Performs a forward pass during action sampling (exploration). This method is called 
        by RLlib to get the output of the network to (non-deterministically) determine the  
        next actions. It returns action distribution inputs (logits/parameters), which RLlib will 
        use to create a stochastic distribution (e.g., softmax, ...) to sample actions from.
        Since action distributions are handled by RLlib internally, this method has the same 
        functional behavior as _forward_inference() such that it simply delegates to it.

        Args:
            batch (Dict[str, Any]): Batch dictionary containing at least the following columns:
                - Columns.OBS (torch.Tensor): Input observations.
                    Shape: (B, obs_dim) or (B, seq_len, obs_dim)
                - Columns.STATE_IN (Dict[str, torch.Tensor]): Hidden states inputs for all RNN layers.
                    Shape: Dict of {layer_name: (B, num_layers*num_directions, hidden_size)}
        
        Returns:
            result (Dict[str, Any]): Dictionary containing the output of the network:
                - Columns.ACTION_DIST_INPUTS (torch.Tensor): Action distribution inputs.
                    Shape: (B, action_dim) or (B, seq_len, action_dim)
                - Columns.STATE_OUT (Dict[str, torch.Tensor]): Hidden states output for all RNN layers.
                    Shape: Dict or single of (num_layers*num_directions, B, hidden_size)
        """
        
        return self._forward_inference(batch)

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Performs a forward pass during training. This method is called by RLlib
        to get the output of the network for computing the loss during training.
        
        Args:
            batch (Dict[str, Any]): Batch dictionary containing at least the following columns:
                - Columns.OBS (torch.Tensor): Input observations.
                    Shape: (B, obs_dim) or (B, seq_len, obs_dim)
                - Columns.STATE_IN (Dict[str, torch.Tensor]): Hidden states inputs for all RNN layers.
                    Shape: Dict of {layer_name: (B, num_layers*num_directions, hidden_size)}
        
        Returns:
            result (Dict[str, Any]): Dictionary containing the output of the network:
                - Columns.ACTION_DIST_INPUTS (torch.Tensor): Action distribution inputs.
                    Shape: (B, action_dim) or (B, seq_len, action_dim)
                - Columns.EMBEDDINGS (torch.Tensor): Embeddings from shared layers.
                    Shape: (B, shared_dim) or (B, seq_len, shared_dim)
                - Columns.STATE_OUT (Dict[str, torch.Tensor]): Hidden states output for all RNN layers.
                    Shape: Dict or single of (num_layers*num_directions, B, hidden_size)
        """

         # Extract observation
        obs = batch.get(Columns.OBS) # Shape: (B, obs_dim) or (B, seq_len, obs_dim)
        if obs is None:
            raise ValueError("Missing Columns.OBS in batch")
        local_obs = obs.get("local")
        global_obs = obs.get("global")
        if local_obs is None or global_obs is None:
            raise ValueError("Expected 'local' and 'global' keys in observation dict")

        # Extract hidden states from batch
        states_in = batch.get(Columns.STATE_IN, {}) # Shape: {} or {layer_name: (B, num_layers*num_directions, hidden_size)}

        # Forward through shared layers if existing
        shared_states_out = None
        embeddings = None
        if self.has_shared_layers:
            shared_hidden = states_in.get("shared_h") # Shape: None or {layer_name: (B, num_layers*num_directions, hidden_size)} 
            shared_out, shared_states_out = self._forward_shared(local_obs, shared_hidden) # Shape: (B, shared_dim) or (B, seq_len, shared_dim), None or {layer_name: (B, num_layers*num_directions, hidden_size)}
            local_obs = shared_out
            embeddings = global_obs if self.use_centralized_critic else shared_out

        # Forward through actor network
        actor_hidden = states_in.get("actor_h") # Shape: None or {layer_name: (B, num_layers*num_directions, hidden_size)}
        actor_out, actor_states_out = self._forward_actor(local_obs, actor_hidden) # Shape: (B, action_dim) or (B, seq_len, action_dim), None or {layer_name: (B, num_layers*num_directions, hidden_size)}

        # Forward through critic network
        critic_obs = global_obs if self.use_centralized_critic else local_obs
        critic_hidden = states_in.get("critic_h") # Shape: None or {layer_name: (B, num_layers*num_directions, hidden_size)}
        critic_out, critic_states_out = self._forward_critic(critic_obs, critic_hidden) # Shape: (B, 1) or (B, seq_len, 1), None or {layer_name: (B, num_layers*num_directions, hidden_size)}

        # Create final states_out dictionary
        states_out = dict(states_in)
        if shared_states_out is not None:
            states_out["shared_h"] = shared_states_out
        if actor_states_out is not None:  
            states_out["actor_h"] = actor_states_out
        if critic_states_out is not None: 
            states_out["critic_h"] = critic_states_out

        # Create final result dictionary
        result = {
            Columns.ACTION_DIST_INPUTS: actor_out
        }
        if embeddings is not None:
            result[Columns.EMBEDDINGS] = embeddings 
        if states_out:
            result[Columns.STATE_OUT] = states_out

        return result

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, Any], embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes value function estimates from observations. This method is called
        by RLlib to get value estimates for computing advantages and value function 
        loss during training.
        
        Args:
            batch (Dict[str, Any]): Batch dictionary containing at least the following columns:
                - Columns.OBS (torch.Tensor): Input observations.
                    Shape: (B, obs_dim) or (B, seq_len, obs_dim)
                - Columns.STATE_IN (Dict[str, torch.Tensor]): Hidden states inputs for all RNN layers.
                    Shape: Dict of {layer_name: (B, num_layers*num_directions, hidden_size)}
            embeddings (Optional[torch.Tensor]): Optional pre-computed embeddings from shared layers.
                Shape: (B, shared_dim) or (B, seq_len, shared_dim)
            
        Returns:
            values (torch.Tensor): Value estimates tensor.
                Shape: (B) or (B, seq_len) (squeezed from (B, 1) or (B, seq_len, 1))
        """

         # Extract observation
        obs = batch.get(Columns.OBS) # Shape: (B, obs_dim) or (B, seq_len, obs_dim)
        if obs is None:
            raise ValueError("Missing Columns.OBS in batch")
        local_obs = obs.get("local")
        global_obs = obs.get("global")
        if local_obs is None or global_obs is None:
            raise ValueError("Expected 'local' and 'global' keys in observation dict")

        # Extract hidden states from batch
        states_in = batch.get(Columns.STATE_IN, {}) # Shape: Dict or single of (B, num_layers*num_directions, hidden_size)

        # If embeddings are provided, use them directly
        if embeddings is not None:
            critic_obs = embeddings
        # If centralized critic is used, use global observation
        elif self.use_centralized_critic:
            critic_obs = global_obs
        # If shared layers are present, forward through shared layers to get embeddings
        elif self.has_shared_layers:
            shared_hidden = states_in.get("shared_h") # Shape: (B, num_layers*num_directions, hidden_size)
            critic_obs, _ = self._forward_shared(local_obs, shared_hidden) # Shape: (B, shared_dim) or (B, seq_len, shared_dim)
        # If no embeddings are provided and no shared layers are present, use local observation
        else:
            critic_obs = local_obs
        
        # Forward through critic network
        critic_hidden = states_in.get("critic_h") # Shape: (B, num_layers*num_directions, hidden_size)
        values, _ = self._forward_critic(critic_obs, critic_hidden) # Shape: (B, 1) or (B, seq_len, 1)
        
        # Squeeze last dimension if it is 1
        if values.dim() > 1 and values.shape[-1] == 1:
            values = values.squeeze(-1) # Shape: (B) or (B, seq_len)
        
        return values
        