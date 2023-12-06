import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from infras.randutils import *
from infras.misc import * 
    
class QueryBandit:
    
    def __init__(self, env, noisy=0.0):
        
        self.env = env
        
        self.dim_S = self.env.dim_S
        self.dim_A = self.env.dim_A

        self.A = self.env.sample_actions()
    
        self.noisy = noisy
        
        self.hist_states = []
        self.hist_actions = []
        self.hist_rewards = []

    def lookup(self,):
        
        stack_states = np.vstack(self.hist_states)
        stack_actions = np.vstack(self.hist_actions)
        stack_rewards = np.vstack(self.hist_rewards)
        
#         print(stack_states.shape)
#         print(stack_actions.shape)
#         print(stack_rewards.shape)
        
        return stack_states, stack_actions, stack_rewards

    def render(self, states, actions, noisy=False):
        
        rewards = self.env.reward_query(states, actions)

        if noisy:
            rewards += np.random.normal(loc=0.0, scale=self.noisy, size=rewards.shape)
            
        if states.ndim == 1:
            states = states.reshape([1,-1])
            
        if actions.ndim == 1:
            actions = actions.reshape([1,-1])
        #
        
        rewards = rewards.reshape([1,-1])

        self.hist_states.append(states)
        self.hist_actions.append(actions)
        self.hist_rewards.append(rewards)

        return rewards
    
    
class BatchBandit:
    
    def __init__(self, env, noisy=0.0):
        
        self.env = env
        
        self.dim_S = self.env.dim_S
        self.dim_A = self.env.dim_A

        self.A = self.env.sample_actions()
    
        self.noisy = noisy
        
        self.hist_states = []
        self.hist_actions = []
        self.hist_rewards = []
        

    def lookup(self,):
        
        stack_states = np.concatenate(self.hist_states, 0)
        stack_actions = np.concatenate(self.hist_actions, 0)
        stack_rewards = np.concatenate(self.hist_rewards, 0)
        
#         print(stack_states.shape)
#         print(stack_actions.shape)
#         print(stack_rewards.shape)
        
        return stack_states, stack_actions, stack_rewards

    def render(self, states, actions, noisy=False):
        
        rewards = self.env.reward_batch(states, actions)

        if noisy:
            rewards += np.random.normal(loc=0.0, scale=self.noisy, size=rewards.shape)
            
        if states.ndim == 1:
            states = states.reshape([1,-1])
            
        if actions.ndim != 3:
            actions = np.expand_dims(actions, 0)
        #

        
        rewards = rewards.reshape([1,-1])

        self.hist_states.append(states)
        self.hist_actions.append(actions)
        self.hist_rewards.append(rewards)

        return rewards
    
class StepBandit:
    
    def __init__(self, env, noisy=0.0):
        
        self.env = env
        
        self.dim_S = self.env.dim_S
        self.dim_A = self.env.dim_A

        self.A = self.env.sample_actions()
    
        self.noisy = noisy
        
        self.hist_states = []
        self.hist_actions = []
        self.hist_rewards = []
        

    def lookup(self,):
        
        stack_states = np.concatenate(self.hist_states, 0)
        stack_actions = np.concatenate(self.hist_actions, 0)
        stack_rewards = np.concatenate(self.hist_rewards, 0)
        
#         print(stack_states.shape)
#         print(stack_actions.shape)
#         print(stack_rewards.shape)
        
        return stack_states, stack_actions, stack_rewards

    def render(self, states, actions, noisy=False):
        
        rewards = self.env.reward_batch(states, actions)

        if noisy:
            rewards += np.random.normal(loc=0.0, scale=self.noisy, size=rewards.shape)
            
        if states.ndim == 1:
            states = states.reshape([1,-1])
            
        if actions.ndim != 3:
            actions = np.expand_dims(actions, 0)
        #

        
        rewards = rewards.reshape([1,-1])

        self.hist_states.append(states)
        self.hist_actions.append(actions)
        self.hist_rewards.append(rewards)

        return rewards
    
    def render_step(self, states, actions, noisy=False):
        
        rewards = self.env.reward_step(states, actions)

        if noisy:
            rewards += np.random.normal(loc=0.0, scale=self.noisy, size=rewards.shape)
            
        return rewards
    
    def update_steps(self, states, actions, rewards):
        
        if states.ndim == 1:
            states = states.reshape([1,-1])
            
        if actions.ndim != 3:
            actions = np.expand_dims(actions, 0)
        #

        
        rewards = rewards.reshape([1,-1])

        self.hist_states.append(states)
        self.hist_actions.append(actions)
        self.hist_rewards.append(rewards)
        

class PINNBandit:
    
    def __init__(self, env, noisy=0.0):
        
        self.env = env
        
        self.dim_S = self.env.dim_S

        self.noisy = noisy
        
        self.hist_states = []
        self.hist_rewards = []

    def render(self, states, noisy=False):
        
        rewards = self.env.reward_query(states)

        if noisy:
            rewards += np.random.normal(loc=0.0, scale=self.noisy, size=rewards.shape)
            
        if states.ndim == 1:
            states = states.reshape([1,-1])
        
        rewards = rewards.reshape([1,-1])

        self.hist_states.append(states)
        self.hist_rewards.append(rewards)

        return rewards
    