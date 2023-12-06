import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm, trange

import warnings

from infras.randutils import *
from infras.misc import *
# from infras.utils import *

from models.gp_v3 import GPR

# from data.games import GamePoisson
# from data.bandits import PDEBandit

# from infras.meters import BanditMeters
# from core.agent import BanditAgent


# from data.pde_poisson import XPINN_Client
# from data.games import GamePoisson
# # from data.bandits import QueryBandit
# from data.bandits import StepBandit


class StepBanditAgent_v3:
    
    def __init__(self, bandit, play_meters, roll_meters, state_dict=None, device=None):
        
        self.device = device
        
        self.bandit = bandit
        self.play_meters = play_meters
        self.roll_meters = roll_meters
        
        self.opts = self.bandit.env.opts

        self._init_models(state_dict)


    def _init_models(self, state_dict):
        self.models = []
        for i in range(self.opts):
            if i == 0:
                self.models.append(GPR(kernel='Hybrid', dim_cont1=self.bandit.env.dim_S).to(self.device))
            else:
                self.models.append(
                    GPR(
                        kernel='Hybrid3', 
                        dim_cont1=1, 
                        dim_cont2=self.bandit.env.dim_S,
                        dim_cat1=self.bandit.env.dim_A,
                        dim_cat2=self.bandit.env.dim_A,
                    ).to(self.device)
                )
            #
        #
        
    def _init_model_idx(self, idx):

        if idx == 0:
            new_model = GPR(kernel='Hybrid', dim_cont1=self.bandit.env.dim_S).to(self.device)
        else:
            new_model = GPR(
                        kernel='Hybrid3', 
                        dim_cont1=1, 
                        dim_cont2=self.bandit.env.dim_S,
                        dim_cat1=self.bandit.env.dim_A,
                        dim_cat2=self.bandit.env.dim_A,
                    ).to(self.device)
        #
        
        self.models[idx] = new_model
        
        
    def replay(self, idx, gam=0.9):
        
        hist_states, hist_actionTs, hist_rewardTs = self.bandit.lookup()
        
        gam_rewardTs = np.zeros(hist_rewardTs.shape)
        
        for i in range(self.bandit.env.opts):
            
            gam_mask = np.power(gam, np.arange(self.opts-i))
            gam_mask = np.vstack([gam_mask]*gam_rewardTs.shape[0])
            
            rewards = hist_rewardTs[:,i:]
            
            R = (gam_mask*rewards)
            R = np.mean(R, axis=1)
            #R = np.sum(R, axis=1)
 
            gam_rewardTs[:,i] = R
        #
        
        states = hist_states
        actions = hist_actionTs[:,idx,:]
        rewards = gam_rewardTs[:,idx].reshape([-1,1])

        if idx == 0:
            X = torch.tensor(np.hstack([states, actions])).to(self.device)
            y = torch.tensor(rewards).to(self.device)
        else:
            prev_actions = hist_actionTs[:,idx-1,:]
            #cprint('b', prev_actions.shape)
            loss = hist_rewardTs[:,idx-1].reshape([-1,1])
            X = torch.tensor(np.hstack([loss, states, prev_actions, actions])).to(self.device)
            y = torch.tensor(rewards).to(self.device)
            
        #
        
        #print(X.shape)
        #print(y.shape)
        
        return X, y
    
    def _update_model_adam(self, idx):
        
        self._init_model_idx(idx)
        
        optimizer = optim.Adam(self.models[idx].parameters(), lr=1e-3)
        
        X, y = self.replay(idx)
        
        max_epochs=5000
        
        for i in range(max_epochs):
            optimizer.zero_grad()
            loss = self.models[idx].eval_nmllh(X, y)
            loss.backward()
            optimizer.step()
        
        #
        
        
    def _update_model_lbfgs(self, idx):
        
        self._init_model_idx(idx)
        
        optimizer = optim.LBFGS(self.models[idx].parameters(), 
                                lr=0.1, 
                                max_iter=5000,
                                tolerance_grad=1e-9,
                                tolerance_change=1e-8, 
                                history_size=50, 
                                line_search_fn='strong_wolfe'
                               )
        
        X, y = self.replay(idx)
        
        #cprint('r', X.shape)
        #cprint('b', y.shape)
        
        def callback(loss):
            #print(loss)
            if loss == torch.nan:
                raise Exception('nan error')
        
        def closure():
            optimizer.zero_grad()
            loss = self.models[idx].eval_nmllh(X, y)
            loss.backward(retain_graph=True)

            callback(loss)

            return loss
        #
        
        optimizer.step(closure)
        
        
    def _safe_update_trunk_model(self, idx):
        
        max_tries_lbfgs = 2
        max_tries_adam = 2
        
        
        for i in range(max_tries_lbfgs):
            try:
                cprint('b', '{}-th update with lbfgs'.format(i+1))
                self._update_model_lbfgs(idx)
                return
            except Exception as e:
                print(e)
                warnings.warn(
                    "{} try of optimizing model with LBFGS failed.".format(i+1)
                )
            #
        #
            
        for i in range(max_tries_adam):
            try:
                cprint('g', '{}-th update with adam'.format(i+1))
                self._update_model_adam(idx)
                return
            except Exception as e:
                print(e)
                warnings.warn(
                    "{} try of optimizing model with Adam failed.".format(i+1)
                )
            #
        #
        
        raise Exception('Error in updating the models after serveral tires...')
    
    
    def _update_trunk_model(self, idx):
        
        self._safe_update_trunk_model(idx)
    
        
    def _step_rnd(self, state, idx, loss=None, prev_action=None, seed=None):
        #print('rnd')
        
        argmax_aid = generate_random_choice(self.bandit.A.shape[0], 1, seed=seed)
        argmax_action = self.bandit.A[argmax_aid,:].reshape([1,-1])
        
        return argmax_action
    
    def _step_ucb(self, state, idx, loss=None, prev_action=None, c=1.0):
        #print('ucb')
        
        state = torch.tensor(state).reshape([1,-1]).to(self.device)
        actions = torch.tensor(self.bandit.A).to(self.device)
        
        states = state.repeat([actions.shape[0], 1])
        
        if idx > 0:
            loss = torch.tensor(loss).reshape([1,1]).to(self.device)
            prev_action = torch.tensor(prev_action).reshape([1,-1]).to(self.device)
            
            vloss = loss.repeat([actions.shape[0], 1])
            #cprint('y', vloss.shape)
            
            #cprint('g', prev_action.shape)
            vprev_actions = prev_action.repeat([actions.shape[0], 1])
            #cprint('y', vprev_actions.shape)
            
            Xte = torch.hstack([vloss, states, vprev_actions, actions])
            Xtr, ytr = self.replay(idx)
            
            #cprint('g', Xte.shape)
            #cprint('g', Xtr.shape)
            #cprint('g', ytr.shape)
            
        else:
            Xte = torch.hstack([states, actions])
            Xtr, ytr = self.replay(idx)
            
        
#         print(Xte.shape)
#         print(Xtr.shape)
#         print(ytr.shape)
        
        mu, var = self.models[idx].forward(Xtr, ytr, Xte)
        
        ucb = mu + c*torch.sqrt(var)
        
        argmax_aid = torch.argmax(ucb.squeeze()).item()
        argmax_action = self.bandit.A[argmax_aid,:].reshape([1,-1])
        
        return argmax_action
    
    def _step_ts(self, state, idx, loss=None, prev_action=None, ns=20):
        #print('ts')
        
        
        state = torch.tensor(state).reshape([1,-1]).to(self.device)
        actions = torch.tensor(self.bandit.A).to(self.device)
        
        states = state.repeat([actions.shape[0], 1])

        if idx > 0:
            loss = torch.tensor(loss).reshape([1,1]).to(self.device)
            prev_action = torch.tensor(prev_action).reshape([1,-1]).to(self.device)
            
            vloss = loss.repeat([actions.shape[0], 1])
            #cprint('y', vloss.shape)
            
            #cprint('g', prev_action.shape)
            vprev_actions = prev_action.repeat([actions.shape[0], 1])
            #cprint('y', vprev_actions.shape)
            
            Xte = torch.hstack([vloss, states, vprev_actions, actions])
            Xtr, ytr = self.replay(idx)
            
            #cprint('g', Xte.shape)
            #cprint('g', Xtr.shape)
            #cprint('g', ytr.shape)
            
        else:
            Xte = torch.hstack([states, actions])
            Xtr, ytr = self.replay(idx)
            
            
        
        #print(Xte.shape)
        #print(Xtr.shape)
        #print(ytr.shape)
        
        mu, var = self.models[idx].forward(Xtr, ytr, Xte)
        
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
    
    
    def _step_mean(self, state, idx, loss=None, prev_action=None):
        
        state = torch.tensor(state).reshape([1,-1]).to(self.device)
        actions = torch.tensor(self.bandit.A).to(self.device)
        
        states = state.repeat([actions.shape[0], 1])
        
        if idx > 0:
            loss = torch.tensor(loss).reshape([1,1]).to(self.device)
            prev_action = torch.tensor(prev_action).reshape([1,-1]).to(self.device)
            
            vloss = loss.repeat([actions.shape[0], 1])
            #cprint('y', vloss.shape)
            
            #cprint('g', prev_action.shape)
            vprev_actions = prev_action.repeat([actions.shape[0], 1])
            #cprint('y', vprev_actions.shape)
            
            Xte = torch.hstack([vloss, states, vprev_actions, actions])
            Xtr, ytr = self.replay(idx)
            
            #cprint('g', Xte.shape)
            #cprint('g', Xtr.shape)
            #cprint('g', ytr.shape)
            
        else:
            Xte = torch.hstack([states, actions])
            Xtr, ytr = self.replay(idx)
            
#         print(Xte.shape)
#         print(Xtr.shape)
#         print(ytr.shape)
        
        mu, var = self.models[idx].forward(Xtr, ytr, Xte)
        
        argmax_aid = torch.argmax(mu.squeeze()).item()
        argmax_action = self.bandit.A[argmax_aid,:].reshape([1,-1])
        
        return argmax_action
    
    
    def _sample_action_pathway(self, state, method, update):
        
        if update:
            self._init_models(state_dict=None)
            for idx in range(self.opts):
                self._update_trunk_model(idx)
            #
        #

        
        As = []
        Rs = []
        
        loss = None
        prev_action=None
        
        for idx in range(self.opts):
            
            #print(loss)

            if method == 'ts':
                action = self._step_ts(state, idx, loss, prev_action)
            elif method == 'ucb':
                action = self._step_ucb(state, idx, loss, prev_action)
            elif method == 'rnd':
                action = self._step_rnd(state, idx, loss, prev_action)
            elif method == 'mean':
                action = self._step_mean(state, idx, loss, prev_action)
            else:
                raise Exception('Invalid method...')
            
            loss = self.bandit.render_step(state, action)
            prev_action = np.copy(action)
            
            As.append(action)
            Rs.append(loss)
        #
        
        As = np.vstack(As).astype(float)
        Rs = np.vstack(Rs).astype(float)
        
        #cprint('r', As.shape)
        #cprint('r', Rs)
        
        return As, Rs
    
    
    def play(self, nfree, nplay, heuristic):
        
        if len(self.bandit.hist_states) == 0:   
            for i in tqdm(range(nfree), desc='warmup'):
                state = self.bandit.env.sample_states(n_sample=1, seed=i)
                actionT_idx = generate_random_choice(self.bandit.A.shape[0], self.opts, seed=i)
                actionT = self.bandit.A[actionT_idx, :]
                rewardT = self.bandit.render(state, actionT)
                #print(state, actionT)
                #cprint('r', rewardT)
                #print(self.bandit.env.scaler_S.inverse_transform(state))
            #
        #
        
#         X,y = self.replay(idx=1)
#         cprint('r', X.shape)
#         cprint('r', y.shape)
        
#         actionT, rewardT = self._sample_action_pathway(state, method='ucb', update=True)
        
        for i in tqdm(range(nplay), desc='play'):
            
            state = self.bandit.env.sample_states(n_sample=1, seed=nfree+i)
            
            if heuristic == 'ts':
                actionT, rewardT = self._sample_action_pathway(state, method='ts', update=True)
            elif heuristic == 'ucb':
                actionT, rewardT = self._sample_action_pathway(state, method='ucb', update=True)
            elif heuristic == 'explore':
                actionT, rewardT = self._sample_action_pathway(state, method='rnd', update=True)
            elif heuristic == 'rnd':
                actionT, rewardT = self._sample_action_pathway(state, method='rnd', update=True)
            else:
                raise Exception('Invalid choice of heuristic...')
            
            state = state.reshape([1,-1])
 
            self.bandit.update_steps(state, actionT, rewardT )
    
            self.play_meters.update(state, actionT, rewardT)
            self.play_meters.save_model_state(self.models)
        #
    
    def roll(self, nroll, heuristic):
        
        for i in tqdm(range(nroll), desc='roll'):
            
            state = self.bandit.env.sample_states(n_sample=1, seed=10000+i)
            
            if heuristic == 'rnd':
                actionT, rewardT = self._sample_action_pathway(state, method='rnd', update=False)
            else:
                actionT, rewardT = self._sample_action_pathway(state, method='mean', update=False)
        
            self.roll_meters.update(state, actionT, rewardT)
        #
        
    def load_bandit_hist(self, hist_data):
        
        hist_states = hist_data['hist_states']
        hist_actions = hist_data['hist_actions']
        hist_rewards = hist_data['hist_rewards']
        
#         print(hist_data['hist_states'].shape)
#         print(hist_data['hist_actions'].shape)
#         print(hist_data['hist_rewards'].shape)
        
        assert hist_states.shape[0] == hist_actions.shape[0] == hist_rewards.shape[0]
        
#         print(hist_states)
        
        hist_states = self.bandit.env.scaler_S.transform(hist_states)
        hist_rewards = -np.log10(hist_rewards)

        nplay = hist_states.shape[0]
        
        hist_states_list = [hist_states[i,:].reshape([1,-1]) for i in range(nplay)]
        hist_rewards_list = [hist_rewards[i,:].reshape([1,-1]) for i in range(nplay)]
        hist_actions_list = [np.expand_dims(hist_actions[i,:,:],0) for i in range(nplay)]
        
        
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

        self._init_models(state_dict=None)
        for idx in range(self.opts):
            self._update_trunk_model(idx)
        #
        
        
        