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

from models.gp import GPR

# from data.games import GamePoisson
# from data.bandits import PDEBandit

# from infras.meters import BanditMeters
# from core.agent import BanditAgent


# from data.pde_poisson import XPINN_Client
# from data.games import GamePoisson
# from data.bandits import QueryBandit

class QueryBanditAgent:
    
    def __init__(self, bandit, play_meters, roll_meters, state_dict=None, device=None):
        
        self.device = device
        
        self.bandit = bandit
        self.play_meters = play_meters
        self.roll_meters = roll_meters
        
        if state_dict is None:
            self.model = GPR(dim_cont=self.bandit.env.dim_S).to(self.device)
        else:
            self.model = GPR(dim_cont=self.bandit.env.dim_S).to(self.device)
        #
        
    def replay(self,):
        
        hist_states, hist_actions, hist_rewards = self.bandit.lookup()
        
        #cprint('b', hist_states.shape)
        #cprint('b', hist_actions.shape)
        #cprint('b', hist_rewards.shape)
        
        X = torch.tensor(np.hstack([hist_states, hist_actions])).to(self.device)
        y = torch.tensor(hist_rewards).to(self.device)
        
        return X, y
    
    def _update_model_adam(self,):
        
        self.model = GPR(dim_cont=self.bandit.env.dim_S).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        
        X, y = self.replay()
        
        max_epochs=5000
        
        for i in range(max_epochs):
            optimizer.zero_grad()
            loss = self.model.eval_nmllh(X, y)
            loss.backward()
            optimizer.step()
        
        #
        
        
    def _update_model_lbfgs(self,):
        
        self.model = GPR(dim_cont=self.bandit.env.dim_S).to(self.device)
        
        optimizer = optim.LBFGS(self.model.parameters(), 
                                lr=0.1, 
                                max_iter=5000,
                                tolerance_grad=1e-9,
                                tolerance_change=1e-8, 
                                history_size=50, 
                                line_search_fn='strong_wolfe'
                               )
        
        X, y = self.replay()
        
        #cprint('r', X.shape)
        #cprint('r', y.shape)
        
        def callback(loss):
            #print(loss)
            if loss == torch.nan:
                raise Exception('nan error')
        
        def closure():
            optimizer.zero_grad()
            loss = self.model.eval_nmllh(X, y)
            loss.backward(retain_graph=True)

            callback(loss)

            return loss
        #
        
        optimizer.step(closure)
        
        
    def _safe_update_model(self, ):
        
        max_tries_lbfgs = 2
        max_tries_adam = 2
        
        
        for i in range(max_tries_lbfgs):
            try:
                self._update_model_lbfgs()
                return
            except:
                warnings.warn(
                    "{} try of optimizing model with LBFGS failed.".format(i+1)
                )
            #
        #
            
        for i in range(max_tries_adam):
            try:
                self._update_model_adam()
                return
            except:
                warnings.warn(
                    "{} try of optimizing model with Adam failed.".format(i+1)
                )
            #
        #
        
        raise Exception('Error in updating the models after serveral tires...')
        

    
    def _update_model(self,):
        
        self._safe_update_model()
        
        
    def _step_rnd(self, state, seed=None):
        #print('rnd')
        
        argmax_aid = generate_random_choice(self.bandit.A.shape[0], 1, seed=seed)
        argmax_action = self.bandit.A[argmax_aid,:].reshape([1,-1])
        
        return argmax_action
    
    def _step_ucb(self, state, c=1.0):
        #print('ucb')
        
        state = torch.tensor(state).reshape([1,-1]).to(self.device)
        actions = torch.tensor(self.bandit.A).to(self.device)
        
        states = state.repeat([actions.shape[0], 1])
        
        Xte = torch.hstack([states, actions])
        Xtr, ytr = self.replay()
        
        mu, var = self.model.forward(Xtr, ytr, Xte)
        
        ucb = mu + c*torch.sqrt(var)
        
        argmax_aid = torch.argmax(ucb.squeeze()).item()
        argmax_action = self.bandit.A[argmax_aid,:].reshape([1,-1])
        
        return argmax_action
    
    def _step_ts(self, state, ns=20):
        #print('ts')
        
        state = torch.tensor(state).reshape([1,-1]).to(self.device)
        actions = torch.tensor(self.bandit.A).to(self.device)
        
        states = state.repeat([actions.shape[0], 1])
        
        Xte = torch.hstack([states, actions])
        Xtr, ytr = self.replay()
        
        mu, var = self.model.forward(Xtr, ytr, Xte)
        
        dist_rewards = torch.distributions.MultivariateNormal(
            loc=mu.squeeze(),
            covariance_matrix=torch.diag(var.squeeze()),
        )
        
        rewards_samples = dist_rewards.sample([ns])
        
#         samples_mean = rewards_samples.mean(0)
#         argmax_aid = torch.argmax(samples_mean).item()

        samples_max, _ = rewards_samples.max(0)
        argmax_aid = torch.argmax(samples_max).item()
        
        argmax_action = self.bandit.A[argmax_aid,:].reshape([1,-1])

        return argmax_action
    
    
    def _step_mean(self, state):
        
        state = torch.tensor(state).reshape([1,-1]).to(self.device)
        actions = torch.tensor(self.bandit.A).to(self.device)
        
        states = state.repeat([actions.shape[0], 1])
        
        Xte = torch.hstack([states, actions])
        Xtr, ytr = self.replay()
        
        mu, var = self.model.forward(Xtr, ytr, Xte)
        
        argmax_aid = torch.argmax(mu.squeeze()).item()
        argmax_action = self.bandit.A[argmax_aid,:].reshape([1,-1])
        
        return argmax_action
        
        
    def play(self, nfree, nplay, heuristic):
        
        if len(self.bandit.hist_states) == 0:   
            for i in tqdm(range(nfree), desc='warmup'):
                state = self.bandit.env.sample_states(n_sample=1, seed=i)
                aid = generate_random_choice(self.bandit.A.shape[0], 1, seed=i)
                action = self.bandit.A[aid, :]
                self.bandit.render(state, action)
                #print(state, action)
                #print(self.bandit.env.scaler_S.inverse_transform(state))
            #
        #
        
        for i in tqdm(range(nplay), desc='play'):

            self._update_model()
            
            state = self.bandit.env.sample_states(n_sample=1, seed=nfree+i)

            #action = self._step_rnd(state, seed=nfree+i)
            #action = self._step_ucb(state)
            #action = self._step_ts(state)
            #action = self._step_mean(state)
            
            if heuristic == 'ts':
                action = self._step_ts(state)
            elif heuristic == 'ucb':
                action = self._step_ucb(state)
            elif heuristic == 'explore':
                action = self._step_rnd(state, seed=nfree+i)
            elif heuristic == 'rnd':
                action = self._step_rnd(state, seed=nfree+i)
            else:
                raise Exception('Invalid choice of heuristic...')
            
            state = state.reshape([1,-1])
 
            r = self.bandit.render(state, action)

            self.play_meters.update(state, action, r)
            self.play_meters.save_model_state(self.model)
        #
    
    def roll(self, nroll, heuristic):
        
        for i in tqdm(range(nroll), desc='roll'):
            
            state = self.bandit.env.sample_states(n_sample=1, seed=10000+i)
            
            if heuristic == 'rnd':
                action = self._step_rnd(state, seed=10000+i)
            else:
                action = self._step_mean(state)
    
            r = self.bandit.render(state, action)
        
            self.roll_meters.update(state, action, r)
        #
        
        
    def load_bandit_hist(self, hist_data):
        
        hist_states = hist_data['hist_states']
        hist_actions = hist_data['hist_actions']
        hist_rewards = hist_data['hist_rewards']
        
#         print(hist_data['hist_states'].shape)
#         print(hist_data['hist_actions'].shape)
#         print(hist_data['hist_rewards'].shape)
        
        assert hist_states.shape[0] == hist_actions.shape[0] == hist_rewards.shape[0]
        
        hist_states = self.bandit.env.scaler_S.transform(hist_states)
        hist_rewards = -np.log10(hist_rewards)

        nplay = hist_states.shape[0]
        
        hist_states_list = [hist_states[i,:].reshape([1,-1]) for i in range(nplay)]
        hist_rewards_list = [hist_rewards[i,:].reshape([1,-1]) for i in range(nplay)]
        hist_actions_list = [hist_actions[i,:].reshape([1,-1]) for i in range(nplay)]
        
        
#         for state in hist_states_list:
#             print(state, state.shape)

#         for r in hist_rewards_list:
#             print(r, r.shape)

#         for a in hist_actions_list:
#             print(a, a.shape)

        self.bandit.hist_states = hist_states_list
        self.bandit.hist_actions = hist_actions_list
        self.bandit.hist_rewards = hist_rewards_list
        
#         X, y = self.replay(idx=1)
#         print(X.shape)
#         print(y.shape)

        self._update_model()
        
        

        
        