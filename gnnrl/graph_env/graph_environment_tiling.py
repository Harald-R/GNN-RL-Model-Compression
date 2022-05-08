import os
import shutil
import torch

import numpy as np
class TilingGraphEnv:

    def __init__(self, graph, n_layers, max_dim_tiles, max_timesteps, log_dir, memory_size_bytes = 917504):
        self.graph = graph
        self.n_layers = n_layers
        self.max_dim_tiles = max_dim_tiles

        self.state = None
        self.done = False
        self.best_reward = 0
        self.best_graph_tiling_scheme = None
        self.max_timesteps = max_timesteps

        self.memory_size_bytes = memory_size_bytes

        self.log_dir = log_dir

    def reset(self):
        self.done = False
        self.best_reward = 0
        self.best_graph_tiling_scheme = None
        self.state = self.graph
        return self.state

    def step(self, action, time_step):

        reward, graph_tiling_scheme = self.compute_reward(action)

        # if reduced_flops >= self.desired_flops:
        #     self.done = True
        #     if self.dataset == "cifar10":
        #         rewards, accuracy,_,_ = reward_caculation(self.pruned_model, self.val_loader, self.device )
        #     else:
        #         _,_,rewards, accuracy = reward_caculation(self.pruned_model, self.val_loader, self.device )

        #     if accuracy > self.best_accuracy:
        #         self.best_accuracy = accuracy

        #         self.save_checkpoint({
        #             'model': self.model_name,
        #             'dataset': self.dataset,
        #             'preserve_ratio':self.preserve_ratio,
        #             'state_dict': self.pruned_model.module.state_dict() if isinstance(self.pruned_model, nn.DataParallel) else self.pruned_model.state_dict(),
        #             'acc': self.best_accuracy,
        #             'flops':r_flops
        #         }, True, checkpoint_dir=self.log_dir)

        #         print("Best Accuracy (without fine-tuning) of Compressed Models: {}. The FLOPs ratio: {}".format( self.best_accuracy,r_flops))

        if time_step == self.max_timesteps:
            if not self.done:
                reward = -100
                self.done = True

        self.update_state(action)

        if reward > self.best_reward:
            self.best_reward = reward
            self.best_graph_tiling_scheme = graph_tiling_scheme
            print('Reward: {}, tiling scheme: {}'.format(self.best_reward, self.best_graph_tiling_scheme))

        return self.state, reward, self.done

    def compute_reward(self, action):
        negative_reward = 0
        positive_reward = 0

        max_layer_positive_reward = 20

        graph_tiling_scheme = []

        for i, layer in enumerate(self.state['x']):
            layer_action = action[i * self.max_dim_tiles : i * self.max_dim_tiles + self.max_dim_tiles]
            new_tiling_scheme = np.argmax(layer_action) + 1

            graph_tiling_scheme.append(new_tiling_scheme)

            layer_size = self.compute_layer_size(layer, new_tiling_scheme)
            if layer_size <= self.memory_size_bytes:
                memory_usage_ratio = (layer_size / self.memory_size_bytes).item()
                positive_reward += memory_usage_ratio * max_layer_positive_reward
            else:
                negative_reward -= 1

        # print('negative_reward:', negative_reward)
        # print('positive_reward:', positive_reward)

        if negative_reward < 0:
            return negative_reward, graph_tiling_scheme
        return positive_reward, graph_tiling_scheme


    def compute_layer_size(self, layer, new_tiling_scheme):
        in_C                   = layer[0]
        in_H                   = layer[1]
        in_W                   = layer[2]
        in_elem_byte_size      = layer[3]
        weights_OC             = layer[4]
        weights_IC             = layer[5]
        weights_KH             = layer[6]
        weights_KW             = layer[7]
        weights_elem_byte_size = layer[8]
        out_C                  = layer[9]
        out_H                  = layer[10]
        out_W                  = layer[11]
        out_elem_byte_size     = layer[12]

        height_tiles = new_tiling_scheme

        input_size = in_C * (in_H / height_tiles) * in_W * in_elem_byte_size
        weights_size = weights_OC * weights_IC * weights_KH * weights_KW * weights_elem_byte_size
        output_size = out_C * (out_H / height_tiles) * out_W * out_elem_byte_size

        return input_size + weights_size + output_size

    def update_state(self, action):
        for i, layer in enumerate(self.state['x']):
            layer_action = action[i * self.max_dim_tiles : i * self.max_dim_tiles + self.max_dim_tiles]
            new_tiling_scheme = np.argmax(layer_action) + 1

            new_layer_features = layer.clone()
            new_layer_features[13] = new_tiling_scheme

            self.state['x'][i] = new_layer_features

    def save_checkpoint(self,state, is_best, checkpoint_dir='.'):
        filename = os.path.join(checkpoint_dir, self.model_name+'ckpt.pth.tar')
        print('=> Saving checkpoint to {}'.format(filename))
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, filename.replace('.pth.tar', '.best.pth.tar'))
