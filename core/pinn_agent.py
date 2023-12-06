import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

import warnings

from tqdm.auto import tqdm, trange

from infras.randutils import *
from infras.misc import *
# from infras.utils import *


class PINNBanditAgent:
    
    def __init__(self, bandit, play_meters, roll_meters, state_dict=None, device=None):
        
        self.device = device
        
        self.bandit = bandit
        self.play_meters = play_meters
        self.roll_meters = roll_meters

        
    def play(self, nfree, nplay):
        
        if len(self.bandit.hist_states) == 0:   
            for i in tqdm(range(nfree), desc='warmup'):
                state = self.bandit.env.sample_states(n_sample=1, seed=i)
                self.bandit.render(state)
                #print(state, action)
                #print(self.bandit.env.scaler_S.inverse_transform(state))
            #
        #
        
        for i in tqdm(range(nplay), desc='play'):
            
            state = self.bandit.env.sample_states(n_sample=1, seed=nfree+i)

            state = state.reshape([1,-1])
 
            r = self.bandit.render(state)

            self.play_meters.update(state, r)
        #
    
    def roll(self, nroll):
        
        for i in tqdm(range(nroll), desc='roll'):
            
            state = self.bandit.env.sample_states(n_sample=1, seed=10000+i)
    
            r = self.bandit.render(state)
        
            self.roll_meters.update(state, r)
        #
        
        