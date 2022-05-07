import os
import shutil
import torch
import torch.nn as nn

from gnnrl.graph_env.graph_construction import hierarchical_graph_construction, net_info
from gnnrl.graph_env.feedback_calculation import reward_caculation
from gnnrl.graph_env.flops_calculation import flops_caculation_forward, preserve_flops
from gnnrl.graph_env.share_layers import share_layer_index
from gnnrl.graph_env.network_pruning import channel_pruning


import numpy as np
import copy

class TilingGraphEnv:

    def __init__(self, graph, n_layers, log_dir, max_timesteps):
        self.graph = graph
        self.n_layers = n_layers

        self.state = None
        self.done = False
        self.max_timesteps = max_timesteps

        self.log_dir = log_dir

    def reset(self):
        self.done = False
        self.state = self.graph
        return self.state

    def step(self, action, time_step):

        # reward = self.compute_reward(action)

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
                rewards = -100
                self.done = True

        self.state = self.update_state(action)

        return self.state, rewards, self.done

    def compute_reward(self, action):
        for layer in self.state:
            # TODO: see if layer fits in memory and determine reward based on that
            pass
        pass

    def update_state(self, action):
        pass

    def save_checkpoint(self,state, is_best, checkpoint_dir='.'):
        filename = os.path.join(checkpoint_dir, self.model_name+'ckpt.pth.tar')
        print('=> Saving checkpoint to {}'.format(filename))
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, filename.replace('.pth.tar', '.best.pth.tar'))
