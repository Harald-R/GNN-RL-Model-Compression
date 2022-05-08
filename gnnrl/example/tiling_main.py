from gnnrl.graph_env.graph_environment_tiling import TilingGraphEnv
from gnnrl.lib.RL.agent_tiling import AgentTiling
from gnnrl.search_tiling import search_tiling

import torch
from torch_geometric.data import Data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def create_graph():
    """
    node_features:
        input channels
        input height
        input width
        input element byte size
        weight output channels
        weight input channels
        weight kernel height
        weight kernel width
        weight element byte size
        tiling scheme
    """
    x = torch.tensor([
        [3, 224, 224, 2, 16, 3, 3, 3, 2, 1],
        [16, 112, 112, 2, 32, 16, 3, 3, 2, 1]
    ], dtype=torch.float, device=device)

    edge_index = torch.tensor([
        [0],
        [1],
    ],dtype=torch.long, device=device)

    num_layers = 2

    return Data(x=x, edge_index=edge_index), num_layers

graph, num_layers = create_graph()

env = TilingGraphEnv(graph,
                     num_layers,
                     log_dir="results_tiling",
                     max_timesteps=5)

max_tiles = 8
num_actions = num_layers * max_tiles

num_features = 10

agent = AgentTiling(state_dim=num_features,
                    action_dim=num_actions,
                    action_std=0.5,
                    lr=0.0003,
                    betas=(0.9, 0.999),
                    gamma=0.99,
                    K_epochs=10,
                    eps_clip=0.2)

search_tiling(env,
              agent,
              update_timestep=100,
              max_timesteps=5,
              max_episodes=15000,
              log_interval=10,
              solved_reward=-10,
              random_seed=None)
