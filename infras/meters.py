import os, sys
import pickle
import logging
import numpy as np
import torch

from infras.misc import *

class QueryBanditMeters(object):
    
    def __init__(self, bandit, save_path, logger=None):
    
        self.save_path = save_path
        create_path(save_path)
        self.logger = logger
        self.bandit = bandit
        
        self.hist_states = []
        self.hist_actions = []
        self.hist_rewards = []
        self.V = 0.0

    def _dump_meter(self,):
        
        res = {}
        
        res['hist_states'] = np.vstack(self.hist_states)
        res['hist_actions'] = np.vstack(self.hist_actions)
        res['hist_rewards'] = np.vstack(self.hist_rewards)
        res['V'] = self.V
        
        pkl_name = 'play{}.pickle'.format(len(self.hist_rewards))

        with open(os.path.join(self.save_path, pkl_name), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        
    def save_model_state(self, model):  
        state_name = 'play{}.dt'.format(len(self.hist_rewards))
        torch.save(model.state_dict(), os.path.join(self.save_path, state_name))
        
    def update(self, s, a, r):
        
        state = s.reshape([1,-1])
        action = a.reshape([1,-1])
        reward = r.reshape([1,-1])
        
        state = self.bandit.env.scaler_S.inverse_transform(state)
        reward = np.power(10, -r)
        
        self.hist_states.append(state)
        self.hist_actions.append(action)
        self.hist_rewards.append(reward)
        
        self.V += np.squeeze(reward)
        
        if self.logger is not None:    
            self.logger.info('===================================')
            self.logger.info('             Play {} '.format(len(self.hist_states)))
            self.logger.info('===================================') 
            
            self.logger.info(' * state  = {} '.format(state))
            self.logger.info(' * action = {} '.format(action))
            self.logger.info(' * reward = {} '.format(reward))
            self.logger.info(' * V      = {} '.format(self.V))

        #
        
        self._dump_meter()
        
class BatchBanditMeters(object):
    
    def __init__(self, bandit, save_path, logger=None):
    
        self.save_path = save_path
        create_path(save_path)
        self.logger = logger
        self.bandit = bandit
        
        self.hist_states = []
        self.hist_actionTs = []
        self.hist_rewardTs = []
        self.V = np.zeros(self.bandit.env.opts)

    def _dump_meter(self,):
        
        res = {}
        
        res['hist_states'] = np.concatenate(self.hist_states, 0)
        res['hist_actions'] = np.concatenate(self.hist_actionTs, 0)
        res['hist_rewards'] = np.concatenate(self.hist_rewardTs, 0)
        res['V'] = self.V
        
        pkl_name = 'play{}.pickle'.format(len(self.hist_states))

        with open(os.path.join(self.save_path, pkl_name), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        
    def save_model_state(self, models):  
        
        for i in range(len(models)):
            state_name = 'play{}_part{}.dt'.format(len(self.hist_states), i)
            torch.save(models[i].state_dict(), os.path.join(self.save_path, state_name))
        
    def update(self, s, aT, rT):
        
        state = s.reshape([1,-1])
        actionT = np.expand_dims(aT, 0)
        rewardT = rT.reshape([1,-1])
        
        
        state = self.bandit.env.scaler_S.inverse_transform(state)
        rewardT = np.power(10, -rewardT)
        
        self.hist_states.append(state)
        self.hist_actionTs.append(actionT)
        self.hist_rewardTs.append(rewardT)
        
        self.V += np.squeeze(rewardT)
        
        if self.logger is not None:    
            self.logger.info('===================================')
            self.logger.info('             Play {} '.format(len(self.hist_states)))
            self.logger.info('===================================') 
            
            self.logger.info(' * state  = {} '.format(state))
            self.logger.info(' * action = {} '.format(actionT))
            self.logger.info(' * reward = {} '.format(rewardT))
            self.logger.info(' * V      = {} '.format(self.V))
        #
        
        self._dump_meter()
        
        
class StepBanditMeters(object):
    
    def __init__(self, bandit, save_path, logger=None):
    
        self.save_path = save_path
        create_path(save_path)
        self.logger = logger
        self.bandit = bandit
        
        self.hist_states = []
        self.hist_actionTs = []
        self.hist_rewardTs = []
        self.V = np.zeros(self.bandit.env.opts)

    def _dump_meter(self,):
        
        res = {}
        
        res['hist_states'] = np.concatenate(self.hist_states, 0)
        res['hist_actions'] = np.concatenate(self.hist_actionTs, 0)
        res['hist_rewards'] = np.concatenate(self.hist_rewardTs, 0)
        res['V'] = self.V
        
        pkl_name = 'play{}.pickle'.format(len(self.hist_states))

        with open(os.path.join(self.save_path, pkl_name), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        
    def save_model_state(self, models):  
        
        for i in range(len(models)):
            state_name = 'play{}_part{}.dt'.format(len(self.hist_states), i)
            torch.save(models[i].state_dict(), os.path.join(self.save_path, state_name))
        
    def update(self, s, aT, rT):
        
        state = s.reshape([1,-1])
        actionT = np.expand_dims(aT, 0)
        rewardT = rT.reshape([1,-1])
        
        
        state = self.bandit.env.scaler_S.inverse_transform(state)
        rewardT = np.power(10, -rewardT)
        
        self.hist_states.append(state)
        self.hist_actionTs.append(actionT)
        self.hist_rewardTs.append(rewardT)
        
        self.V += np.squeeze(rewardT)
        
        if self.logger is not None:    
            self.logger.info('===================================')
            self.logger.info('             Play {} '.format(len(self.hist_states)))
            self.logger.info('===================================') 
            
            self.logger.info(' * state  = {} '.format(state))
            self.logger.info(' * action = {} '.format(actionT))
            self.logger.info(' * reward = {} '.format(rewardT))
            self.logger.info(' * V      = {} '.format(self.V))
        #
        
        self._dump_meter()
        
        
class PINNBanditMeters(object):
    
    def __init__(self, bandit, save_path, logger=None):

        self.save_path = save_path
        create_path(save_path)
        self.logger = logger
        self.bandit = bandit

        self.hist_states = []
        self.hist_rewards = []
        self.V = 0.0
        
        
    def _dump_meter(self,):
        
        res = {}
        
        res['hist_states'] = np.vstack(self.hist_states)
        res['hist_rewards'] = np.vstack(self.hist_rewards)
        res['V'] = self.V
        
        pkl_name = 'play{}.pickle'.format(len(self.hist_rewards))

        with open(os.path.join(self.save_path, pkl_name), 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        
    def save_model_state(self, model):  
        state_name = 'play{}.dt'.format(len(self.hist_rewards))
        torch.save(model.state_dict(), os.path.join(self.save_path, state_name))
        
    def update(self, s, r):
        
        state = s.reshape([1,-1])
        reward = r.reshape([1,-1])
        
        state = self.bandit.env.scaler_S.inverse_transform(state)
        reward = np.power(10, -r)
        
        self.hist_states.append(state)
        self.hist_rewards.append(reward)
        
        self.V += np.squeeze(reward)
        
        if self.logger is not None:    
            self.logger.info('===================================')
            self.logger.info('             Play {} '.format(len(self.hist_states)))
            self.logger.info('===================================') 
            
            self.logger.info(' * state  = {} '.format(state))
            self.logger.info(' * reward = {} '.format(reward))
            self.logger.info(' * V      = {} '.format(self.V))

        #
        
        self._dump_meter()
        
        
        
        
