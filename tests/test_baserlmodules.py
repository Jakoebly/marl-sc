import numpy as np
import gymnasium as gym
import torch
from ray.rllib.core.columns import Columns
import tree
tree_map_structure = tree.map_structure

from src.algorithms.models.rlmodules.ippo_module import IPPORLModule


def test_ippo_module_mlp():
    """Test IPPORLModule with MLP architecture."""
    print("=" * 80)
    print("Testing IPPORLModule with MLP architecture")
    print("=" * 80)
    
    # Configuration parameters
    B = 10  # batch size
    T = 5  # seq len
    n_skus = 10
    obs_dim = 2 * n_skus  # 20
    action_dim = n_skus  # 10
    
    # Model configuration for MLP
    model_config = {
        "networks": {
            "shared_layers": {
                "type": "mlp",
                "output_dim": 256,
                "config": {
                    "hidden_sizes": [256],
                    "activation": "relu"
                }
            },
            "actor": {
                "type": "mlp",
                "config": {
                    "hidden_sizes": [256, 256],
                    "activation": "relu"
                }
            },
            "critic": {
                "type": "mlp",
                "config": {
                    "hidden_sizes": [64, 64],
                    "activation": "relu"
                }
            }
        }
    }
    
    # Construct the RLModule
    my_net = IPPORLModule(
        observation_space=gym.spaces.Box(-1.0, 1.0, (obs_dim,), np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, (action_dim,), np.float32),
        model_config=model_config,
        inference_only=False
    )
    
    print(f"[OK] RLModule created with obs_dim={obs_dim}, action_dim={action_dim}")
    
    # Create dummy input for single timestep (B, obs_dim)
    obs = torch.from_numpy(
        np.random.random_sample(size=(B, obs_dim)).astype(np.float32)
    )
    
    # Get initial state (empty dict for MLP, no hidden states)
    state_in = my_net.get_initial_state()
    print(f"[OK] Initial state: {state_in}")
    
    # Convert state_in to torch tensors and repeat across batch
    state_in = tree_map_structure(
        lambda s: torch.from_numpy(s).unsqueeze(0).repeat(B, 1, 1) if isinstance(s, np.ndarray) else None,
        state_in
    )
    # Filter out None values
    state_in = {k: v for k, v in state_in.items() if v is not None}
    
    input_dict = {
        Columns.OBS: obs,
        Columns.STATE_IN: state_in if state_in else {},
    }
    
    # Run through all 3 forward passes
    print("\n--- Forward Inference ---")
    result_inference = my_net.forward_inference(input_dict)
    print(f"ACTION_DIST_INPUTS shape: {result_inference[Columns.ACTION_DIST_INPUTS].shape}")
    if Columns.STATE_OUT in result_inference:
        print(f"STATE_OUT keys: {result_inference[Columns.STATE_OUT].keys()}")
    
    print("\n--- Forward Exploration ---")
    result_exploration = my_net.forward_exploration(input_dict)
    print(f"ACTION_DIST_INPUTS shape: {result_exploration[Columns.ACTION_DIST_INPUTS].shape}")
    if Columns.STATE_OUT in result_exploration:
        print(f"STATE_OUT keys: {result_exploration[Columns.STATE_OUT].keys()}")
    
    print("\n--- Forward Train ---")
    result_train = my_net.forward_train(input_dict)
    print(f"ACTION_DIST_INPUTS shape: {result_train[Columns.ACTION_DIST_INPUTS].shape}")
    if Columns.EMBEDDINGS in result_train:
        print(f"EMBEDDINGS shape: {result_train[Columns.EMBEDDINGS].shape}")
    if Columns.STATE_OUT in result_train:
        print(f"STATE_OUT keys: {result_train[Columns.STATE_OUT].keys()}")
    
    # Test compute_values
    print("\n--- Compute Values ---")
    values = my_net.compute_values(input_dict)
    print(f"Values shape: {values.shape}")
    print(f"Values sample: {values[:3]}")
    
    # Test compute_values with embeddings
    if Columns.EMBEDDINGS in result_train:
        values_with_embeddings = my_net.compute_values(input_dict, embeddings=result_train[Columns.EMBEDDINGS])
        print(f"Values with embeddings shape: {values_with_embeddings.shape}")
    
    # Print out the number of parameters
    num_all_params = sum(int(np.prod(p.size())) for p in my_net.parameters())
    print(f"\n[OK] Total number of parameters: {num_all_params:,}")
    print()


def test_ippo_module_gru():
    """Test IPPORLModule with GRU architecture."""
    print("=" * 80)
    print("Testing IPPORLModule with GRU architecture")
    print("=" * 80)
    
    # Configuration parameters
    B = 10  # batch size
    T = 5  # seq len
    n_skus = 10
    obs_dim = 2 * n_skus  # 20
    action_dim = n_skus  # 10
    GRU_HIDDEN_SIZE = 128
    GRU_NUM_LAYERS = 2
    
    # Model configuration for GRU
    model_config = {
        "networks": {
            "shared_layers": {
                "type": "gru",
                "output_dim": 256,
                "config": {
                    "hidden_size": GRU_HIDDEN_SIZE,
                    "num_layers": GRU_NUM_LAYERS,
                    "bidirectional": False
                }
            },
            "actor": {
                "type": "gru",
                "config": {
                    "hidden_size": GRU_HIDDEN_SIZE,
                    "num_layers": GRU_NUM_LAYERS,
                    "bidirectional": False
                }
            },
            "critic": {
                "type": "gru",
                "config": {
                    "hidden_size": GRU_HIDDEN_SIZE,
                    "num_layers": GRU_NUM_LAYERS,
                    "bidirectional": False
                }
            }
        }
    }
    
    # Construct the RLModule
    my_net = IPPORLModule(
        observation_space=gym.spaces.Box(-1.0, 1.0, (obs_dim,), np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, (action_dim,), np.float32),
        model_config=model_config
    )
    
    print(f"[OK] RLModule created with obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"[OK] GRU config: hidden_size={GRU_HIDDEN_SIZE}, num_layers={GRU_NUM_LAYERS}")
    
    # Create dummy input for sequence (B, T, obs_dim)
    obs = torch.from_numpy(
        np.random.random_sample(size=(B, T, obs_dim)).astype(np.float32)
    )
    
    # Get initial state (contains hidden states for GRU)
    state_in = my_net.get_initial_state()
    print(f"[OK] Initial state keys: {list(state_in.keys())}")
    
    # Convert state_in to torch tensors and repeat across batch
    # GRU states have shape (num_layers*num_directions, hidden_size)
    # We need to add batch dimension: (B, num_layers*num_directions, hidden_size)
    state_in_torch = {}
    for key, state in state_in.items():
        if isinstance(state, np.ndarray):
            # state is (num_layers*num_directions, hidden_size)
            # Expand to (B, num_layers*num_directions, hidden_size)
            state_tensor = torch.from_numpy(state).unsqueeze(0).repeat(B, 1, 1)
            state_in_torch[key] = state_tensor
            print(f"[OK] {key} shape: {state_tensor.shape}")
    
    input_dict = {
        Columns.OBS: obs,
        Columns.STATE_IN: state_in_torch,
    }
    
    # Run through all 3 forward passes
    print("\n--- Forward Inference ---")
    result_inference = my_net.forward_inference(input_dict)
    print(f"ACTION_DIST_INPUTS shape: {result_inference[Columns.ACTION_DIST_INPUTS].shape}")
    if Columns.STATE_OUT in result_inference:
        for key, state in result_inference[Columns.STATE_OUT].items():
            print(f"STATE_OUT[{key}] shape: {state.shape}")
    
    print("\n--- Forward Exploration ---")
    result_exploration = my_net.forward_exploration(input_dict)
    print(f"ACTION_DIST_INPUTS shape: {result_exploration[Columns.ACTION_DIST_INPUTS].shape}")
    if Columns.STATE_OUT in result_exploration:
        for key, state in result_exploration[Columns.STATE_OUT].items():
            print(f"STATE_OUT[{key}] shape: {state.shape}")
    
    print("\n--- Forward Train ---")
    result_train = my_net.forward_train(input_dict)
    print(f"ACTION_DIST_INPUTS shape: {result_train[Columns.ACTION_DIST_INPUTS].shape}")
    if Columns.EMBEDDINGS in result_train:
        print(f"EMBEDDINGS shape: {result_train[Columns.EMBEDDINGS].shape}")
    if Columns.STATE_OUT in result_train:
        for key, state in result_train[Columns.STATE_OUT].items():
            print(f"STATE_OUT[{key}] shape: {state.shape}")
    
    # Test compute_values
    print("\n--- Compute Values ---")
    values = my_net.compute_values(input_dict)
    print(f"Values shape: {values.shape}")
    print(f"Values sample: {values[0, :3]}")
    
    # Test compute_values with embeddings
    if Columns.EMBEDDINGS in result_train:
        values_with_embeddings = my_net.compute_values(input_dict, embeddings=result_train[Columns.EMBEDDINGS])
        print(f"Values with embeddings shape: {values_with_embeddings.shape}")
    
    # Print out the number of parameters
    num_all_params = sum(int(np.prod(p.size())) for p in my_net.parameters())
    print(f"\n[OK] Total number of parameters: {num_all_params:,}")
    print()


def test_ippo_module_no_shared_layers():
    """Test IPPORLModule without shared layers."""
    print("=" * 80)
    print("Testing IPPORLModule without shared layers")
    print("=" * 80)
    
    # Configuration parameters
    B = 10  # batch size
    n_skus = 10
    obs_dim = 2 * n_skus  # 20
    action_dim = n_skus  # 10
    
    # Model configuration without shared layers
    model_config = {
        "networks": {
            "actor": {
                "type": "mlp",
                "config": {
                    "hidden_sizes": [256, 256],
                    "activation": "relu"
                }
            },
            "critic": {
                "type": "mlp",
                "config": {
                    "hidden_sizes": [64, 64],
                    "activation": "relu"
                }
            }
        }
    }
    
    # Construct the RLModule
    my_net = IPPORLModule(
        observation_space=gym.spaces.Box(-1.0, 1.0, (obs_dim,), np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, (action_dim,), np.float32),
        model_config=model_config
    )
    
    print(f"[OK] RLModule created without shared layers")
    
    # Create dummy input
    obs = torch.from_numpy(
        np.random.random_sample(size=(B, obs_dim)).astype(np.float32)
    )
    
    # Get initial state (should be empty for MLP)
    state_in = my_net.get_initial_state()
    print(f"[OK] Initial state: {state_in}")
    
    input_dict = {
        Columns.OBS: obs,
        Columns.STATE_IN: {},
    }
    
    # Run forward passes
    print("\n--- Forward Inference ---")
    result_inference = my_net.forward_inference(input_dict)
    print(f"ACTION_DIST_INPUTS shape: {result_inference[Columns.ACTION_DIST_INPUTS].shape}")
    
    print("\n--- Forward Train ---")
    result_train = my_net.forward_train(input_dict)
    print(f"ACTION_DIST_INPUTS shape: {result_train[Columns.ACTION_DIST_INPUTS].shape}")
    # Should not have embeddings when no shared layers
    if Columns.EMBEDDINGS in result_train:
        print(f"EMBEDDINGS shape: {result_train[Columns.EMBEDDINGS].shape}")
    else:
        print("No EMBEDDINGS (expected when no shared layers)")
    
    # Test compute_values
    print("\n--- Compute Values ---")
    values = my_net.compute_values(input_dict)
    print(f"Values shape: {values.shape}")
    
    # Print out the number of parameters
    num_all_params = sum(int(np.prod(p.size())) for p in my_net.parameters())
    print(f"\n[OK] Total number of parameters: {num_all_params:,}")
    print()


def test_ippo_module_shared_gru_only():
    """Test IPPORLModule with only shared layers using GRU (actor and critic use MLP)."""
    print("=" * 80)
    print("Testing IPPORLModule with only shared layers using GRU")
    print("=" * 80)
    
    # Configuration parameters
    B = 10  # batch size
    T = 5  # seq len
    n_skus = 10
    obs_dim = 2 * n_skus  # 20
    action_dim = n_skus  # 10
    GRU_HIDDEN_SIZE = 128
    GRU_NUM_LAYERS = 2
    
    # Model configuration: only shared layers use GRU
    model_config = {
        "networks": {
            "shared_layers": {
                "type": "gru",
                "output_dim": 256,
                "config": {
                    "hidden_size": GRU_HIDDEN_SIZE,
                    "num_layers": GRU_NUM_LAYERS,
                    "bidirectional": False
                }
            },
            "actor": {
                "type": "mlp",
                "config": {
                    "hidden_sizes": [256, 256],
                    "activation": "relu"
                }
            },
            "critic": {
                "type": "mlp",
                "config": {
                    "hidden_sizes": [64, 64],
                    "activation": "relu"
                }
            }
        }
    }
    
    # Construct the RLModule
    my_net = IPPORLModule(
        observation_space=gym.spaces.Box(-1.0, 1.0, (obs_dim,), np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, (action_dim,), np.float32),
        model_config=model_config
    )
    
    print(f"[OK] RLModule created with obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"[OK] Shared layers: GRU (hidden_size={GRU_HIDDEN_SIZE}, num_layers={GRU_NUM_LAYERS})")
    print(f"[OK] Actor: MLP, Critic: MLP")
    
    # Create dummy input for sequence (B, T, obs_dim) - sequence needed for GRU
    obs = torch.from_numpy(
        np.random.random_sample(size=(B, T, obs_dim)).astype(np.float32)
    )
    
    # Get initial state (should only have shared_h for GRU)
    state_in = my_net.get_initial_state()
    print(f"[OK] Initial state keys: {list(state_in.keys())}")
    
    # Convert state_in to torch tensors and repeat across batch
    state_in_torch = {}
    for key, state in state_in.items():
        if isinstance(state, np.ndarray):
            # state is (num_layers*num_directions, hidden_size)
            # Expand to (B, num_layers*num_directions, hidden_size)
            state_tensor = torch.from_numpy(state).unsqueeze(0).repeat(B, 1, 1)
            state_in_torch[key] = state_tensor
            print(f"[OK] {key} shape: {state_tensor.shape}")
    
    input_dict = {
        Columns.OBS: obs,
        Columns.STATE_IN: state_in_torch,
    }
    
    # Run through all 3 forward passes
    print("\n--- Forward Inference ---")
    result_inference = my_net.forward_inference(input_dict)
    print(f"ACTION_DIST_INPUTS shape: {result_inference[Columns.ACTION_DIST_INPUTS].shape}")
    if Columns.STATE_OUT in result_inference:
        for key, state in result_inference[Columns.STATE_OUT].items():
            print(f"STATE_OUT[{key}] shape: {state.shape}")
    
    print("\n--- Forward Exploration ---")
    result_exploration = my_net.forward_exploration(input_dict)
    print(f"ACTION_DIST_INPUTS shape: {result_exploration[Columns.ACTION_DIST_INPUTS].shape}")
    if Columns.STATE_OUT in result_exploration:
        for key, state in result_exploration[Columns.STATE_OUT].items():
            print(f"STATE_OUT[{key}] shape: {state.shape}")
    
    print("\n--- Forward Train ---")
    result_train = my_net.forward_train(input_dict)
    print(f"ACTION_DIST_INPUTS shape: {result_train[Columns.ACTION_DIST_INPUTS].shape}")
    if Columns.EMBEDDINGS in result_train:
        print(f"EMBEDDINGS shape: {result_train[Columns.EMBEDDINGS].shape}")
    if Columns.STATE_OUT in result_train:
        for key, state in result_train[Columns.STATE_OUT].items():
            print(f"STATE_OUT[{key}] shape: {state.shape}")
    
    # Test compute_values
    print("\n--- Compute Values ---")
    values = my_net.compute_values(input_dict)
    print(f"Values shape: {values.shape}")
    print(f"Values sample: {values[0, :3]}")
    
    # Test compute_values with embeddings
    if Columns.EMBEDDINGS in result_train:
        values_with_embeddings = my_net.compute_values(input_dict, embeddings=result_train[Columns.EMBEDDINGS])
        print(f"Values with embeddings shape: {values_with_embeddings.shape}")
    
    # Print out the number of parameters
    num_all_params = sum(int(np.prod(p.size())) for p in my_net.parameters())
    print(f"\n[OK] Total number of parameters: {num_all_params:,}")
    print()


def test_ippo_module_actor_gru_only():
    """Test IPPORLModule with only actor using GRU (shared layers and critic use MLP)."""
    print("=" * 80)
    print("Testing IPPORLModule with only actor using GRU")
    print("=" * 80)
    
    # Configuration parameters
    B = 10  # batch size
    T = 5  # seq len
    n_skus = 10
    obs_dim = 2 * n_skus  # 20
    action_dim = n_skus  # 10
    GRU_HIDDEN_SIZE = 128
    GRU_NUM_LAYERS = 2
    
    # Model configuration: only actor uses GRU
    model_config = {
        "networks": {
            "shared_layers": {
                "type": "mlp",
                "output_dim": 256,
                "config": {
                    "hidden_sizes": [256],
                    "activation": "relu"
                }
            },
            "actor": {
                "type": "gru",
                "config": {
                    "hidden_size": GRU_HIDDEN_SIZE,
                    "num_layers": GRU_NUM_LAYERS,
                    "bidirectional": False
                }
            },
            "critic": {
                "type": "mlp",
                "config": {
                    "hidden_sizes": [64, 64],
                    "activation": "relu"
                }
            }
        }
    }
    
    # Construct the RLModule
    my_net = IPPORLModule(
        observation_space=gym.spaces.Box(-1.0, 1.0, (obs_dim,), np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, (action_dim,), np.float32),
        model_config=model_config
    )
    
    print(f"[OK] RLModule created with obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"[OK] Shared layers: MLP")
    print(f"[OK] Actor: GRU (hidden_size={GRU_HIDDEN_SIZE}, num_layers={GRU_NUM_LAYERS})")
    print(f"[OK] Critic: MLP")
    
    # Create dummy input for sequence (B, T, obs_dim) - sequence needed for GRU
    obs = torch.from_numpy(
        np.random.random_sample(size=(B, T, obs_dim)).astype(np.float32)
    )
    
    # Get initial state (should only have actor_h for GRU)
    state_in = my_net.get_initial_state()
    print(f"[OK] Initial state keys: {list(state_in.keys())}")
    
    # Convert state_in to torch tensors and repeat across batch
    state_in_torch = {}
    for key, state in state_in.items():
        if isinstance(state, np.ndarray):
            # state is (num_layers*num_directions, hidden_size)
            # Expand to (B, num_layers*num_directions, hidden_size)
            state_tensor = torch.from_numpy(state).unsqueeze(0).repeat(B, 1, 1)
            state_in_torch[key] = state_tensor
            print(f"[OK] {key} shape: {state_tensor.shape}")
    
    input_dict = {
        Columns.OBS: obs,
        Columns.STATE_IN: state_in_torch,
    }
    
    # Run through all 3 forward passes
    print("\n--- Forward Inference ---")
    result_inference = my_net.forward_inference(input_dict)
    print(f"ACTION_DIST_INPUTS shape: {result_inference[Columns.ACTION_DIST_INPUTS].shape}")
    if Columns.STATE_OUT in result_inference:
        for key, state in result_inference[Columns.STATE_OUT].items():
            print(f"STATE_OUT[{key}] shape: {state.shape}")
    
    print("\n--- Forward Exploration ---")
    result_exploration = my_net.forward_exploration(input_dict)
    print(f"ACTION_DIST_INPUTS shape: {result_exploration[Columns.ACTION_DIST_INPUTS].shape}")
    if Columns.STATE_OUT in result_exploration:
        for key, state in result_exploration[Columns.STATE_OUT].items():
            print(f"STATE_OUT[{key}] shape: {state.shape}")
    
    print("\n--- Forward Train ---")
    result_train = my_net.forward_train(input_dict)
    print(f"ACTION_DIST_INPUTS shape: {result_train[Columns.ACTION_DIST_INPUTS].shape}")
    if Columns.EMBEDDINGS in result_train:
        print(f"EMBEDDINGS shape: {result_train[Columns.EMBEDDINGS].shape}")
    if Columns.STATE_OUT in result_train:
        for key, state in result_train[Columns.STATE_OUT].items():
            print(f"STATE_OUT[{key}] shape: {state.shape}")
    
    # Test compute_values
    print("\n--- Compute Values ---")
    values = my_net.compute_values(input_dict)
    print(f"Values shape: {values.shape}")
    print(f"Values sample: {values[0, :3]}")
    
    # Test compute_values with embeddings
    if Columns.EMBEDDINGS in result_train:
        values_with_embeddings = my_net.compute_values(input_dict, embeddings=result_train[Columns.EMBEDDINGS])
        print(f"Values with embeddings shape: {values_with_embeddings.shape}")
    
    # Print out the number of parameters
    num_all_params = sum(int(np.prod(p.size())) for p in my_net.parameters())
    print(f"\n[OK] Total number of parameters: {num_all_params:,}")
    print()


def test_ippo_module_critic_gru_only():
    """Test IPPORLModule with only critic using GRU (shared layers and actor use MLP)."""
    print("=" * 80)
    print("Testing IPPORLModule with only critic using GRU")
    print("=" * 80)
    
    # Configuration parameters
    B = 10  # batch size
    T = 5  # seq len
    n_skus = 10
    obs_dim = 2 * n_skus  # 20
    action_dim = n_skus  # 10
    GRU_HIDDEN_SIZE = 128
    GRU_NUM_LAYERS = 2
    
    # Model configuration: only critic uses GRU
    model_config = {
        "networks": {
            "shared_layers": {
                "type": "mlp",
                "output_dim": 256,
                "config": {
                    "hidden_sizes": [256],
                    "activation": "relu"
                }
            },
            "actor": {
                "type": "mlp",
                "config": {
                    "hidden_sizes": [256, 256],
                    "activation": "relu"
                }
            },
            "critic": {
                "type": "gru",
                "config": {
                    "hidden_size": GRU_HIDDEN_SIZE,
                    "num_layers": GRU_NUM_LAYERS,
                    "bidirectional": False
                }
            }
        }
    }
    
    # Construct the RLModule
    my_net = IPPORLModule(
        observation_space=gym.spaces.Box(-1.0, 1.0, (obs_dim,), np.float32),
        action_space=gym.spaces.Box(-1.0, 1.0, (action_dim,), np.float32),
        model_config=model_config
    )
    
    print(f"[OK] RLModule created with obs_dim={obs_dim}, action_dim={action_dim}")
    print(f"[OK] Shared layers: MLP")
    print(f"[OK] Actor: MLP")
    print(f"[OK] Critic: GRU (hidden_size={GRU_HIDDEN_SIZE}, num_layers={GRU_NUM_LAYERS})")
    
    # Create dummy input for sequence (B, T, obs_dim) - sequence needed for GRU
    obs = torch.from_numpy(
        np.random.random_sample(size=(B, T, obs_dim)).astype(np.float32)
    )
    
    # Get initial state (should only have critic_h for GRU)
    state_in = my_net.get_initial_state()
    print(f"[OK] Initial state keys: {list(state_in.keys())}")
    
    # Convert state_in to torch tensors and repeat across batch
    state_in_torch = {}
    for key, state in state_in.items():
        if isinstance(state, np.ndarray):
            # state is (num_layers*num_directions, hidden_size)
            # Expand to (B, num_layers*num_directions, hidden_size)
            state_tensor = torch.from_numpy(state).unsqueeze(0).repeat(B, 1, 1)
            state_in_torch[key] = state_tensor
            print(f"[OK] {key} shape: {state_tensor.shape}")
    
    input_dict = {
        Columns.OBS: obs,
        Columns.STATE_IN: state_in_torch,
    }
    
    # Run through all 3 forward passes
    print("\n--- Forward Inference ---")
    result_inference = my_net.forward_inference(input_dict)
    print(f"ACTION_DIST_INPUTS shape: {result_inference[Columns.ACTION_DIST_INPUTS].shape}")
    if Columns.STATE_OUT in result_inference:
        for key, state in result_inference[Columns.STATE_OUT].items():
            print(f"STATE_OUT[{key}] shape: {state.shape}")
    
    print("\n--- Forward Exploration ---")
    result_exploration = my_net.forward_exploration(input_dict)
    print(f"ACTION_DIST_INPUTS shape: {result_exploration[Columns.ACTION_DIST_INPUTS].shape}")
    if Columns.STATE_OUT in result_exploration:
        for key, state in result_exploration[Columns.STATE_OUT].items():
            print(f"STATE_OUT[{key}] shape: {state.shape}")
    
    print("\n--- Forward Train ---")
    result_train = my_net.forward_train(input_dict)
    print(f"ACTION_DIST_INPUTS shape: {result_train[Columns.ACTION_DIST_INPUTS].shape}")
    if Columns.EMBEDDINGS in result_train:
        print(f"EMBEDDINGS shape: {result_train[Columns.EMBEDDINGS].shape}")
    if Columns.STATE_OUT in result_train:
        for key, state in result_train[Columns.STATE_OUT].items():
            print(f"STATE_OUT[{key}] shape: {state.shape}")
    
    # Test compute_values
    print("\n--- Compute Values ---")
    values = my_net.compute_values(input_dict)
    print(f"Values shape: {values.shape}")
    print(f"Values sample: {values[0, :3]}")
    
    # Test compute_values with embeddings
    if Columns.EMBEDDINGS in result_train:
        values_with_embeddings = my_net.compute_values(input_dict, embeddings=result_train[Columns.EMBEDDINGS])
        print(f"Values with embeddings shape: {values_with_embeddings.shape}")
    
    # Print out the number of parameters
    num_all_params = sum(int(np.prod(p.size())) for p in my_net.parameters())
    print(f"\n[OK] Total number of parameters: {num_all_params:,}")
    print()


if __name__ == "__main__":
    # Run all tests
    test_ippo_module_mlp()
    #test_ippo_module_gru()
    #test_ippo_module_no_shared_layers()
    #test_ippo_module_shared_gru_only()
    #test_ippo_module_actor_gru_only()
    #test_ippo_module_critic_gru_only()
    print("=" * 80)
    print("All tests completed!")
    print("=" * 80)