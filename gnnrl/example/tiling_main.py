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

        weights output channels
        weights input channels
        weights kernel height
        weights kernel width
        weights element byte size

        output channels
        output height
        output width
        output element byte size

        tiling scheme
    """
    x = torch.tensor([
        [3, 512, 512, 2, 16, 3, 3, 3, 2, 16, 256, 256, 2, 1],
        [16, 256, 256, 2, 32, 16, 3, 3, 2, 32, 128, 128, 2, 1],
        [32, 128, 128, 2, 64, 32, 3, 3, 2, 64, 64, 64, 2, 1],
        [64, 64, 64, 2, 64, 64, 3, 3, 2, 128, 64, 64, 2, 1],
        [128, 64, 64, 2, 256, 128, 3, 3, 2, 256, 64, 64, 2, 1]
    ], dtype=torch.float, device=device)

    edge_index = torch.tensor([
        [0, 1, 2, 3],
        [1, 2, 3, 4],
    ],dtype=torch.long, device=device)

    num_layers = len(x)

    return Data(x=x, edge_index=edge_index), num_layers

graph, num_layers = create_graph()

max_dim_tiles = 10

env = TilingGraphEnv(graph,
                     num_layers,
                     max_dim_tiles,
                     max_timesteps=5,
                     log_dir="results_tiling")

num_features = 14
num_actions = num_layers * (max_dim_tiles * 2)  # supports tiling over C and H for each layer

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
              update_timestep=500,
              max_timesteps=100,
              max_episodes=15000,
              log_interval=10,
              solved_reward=-10,
              random_seed=None)
