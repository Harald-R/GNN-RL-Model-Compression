from gnnrl.graph_env.graph_environment_tiling import TilingGraphEnv
from gnnrl.lib.RL.agent import Agent
from gnnrl.search_tiling import search_tiling

## TODO:  generate graph
graph = None
n_layers = 2

env = TilingGraphEnv(graph,
                     n_layers,
                     log_dir="results_tiling",
                     max_timesteps=5)

agent = Agent(state_dim=20,
              action_dim=n_layers,
              action_std=0.5,
              lr=0.0003,
              betas=(0.9, 0.999),
              gamma=0.99,
              K_epochs=10,
              eps_clip=0.2,
              plain=True)

search_tiling(env,
              agent,
              update_timestep=100,
              max_timesteps=5,
              max_episodes=15000,
              log_interval=10,
              solved_reward=-10,
              random_seed=None)
