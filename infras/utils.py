import os, sys
import pickle
import logging
import numpy as np
import torch


def get_logger(logpath, displaying=True, saving=True, debug=False, append=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        if append:
            info_file_handler = logging.FileHandler(logpath, mode="a")
        else:
            info_file_handler = logging.FileHandler(logpath, mode="w+")
        #
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    return logger

# class BanditMeters(object):
    
#     def __init__(self, game, save_path, logger):
    
#         self.save_path = save_path
#         self.logger = logger
#         self.game = game
        
#         self.hist_states = []
#         self.hist_actions = []
#         self.hist_rewards = []
#         self.V = 0.0

#     def _dump_meter(self,):
        
#         res = {}
        
#         res['hist_states'] = np.vstack(self.hist_states)
#         res['hist_actions'] = np.vstack(self.hist_actions)
#         res['hist_rewards'] = np.vstack(self.hist_rewards)
#         res['V'] = self.V
        
#         pkl_name = 'play{}.pickle'.format(len(self.hist_rewards))

#         with open(os.path.join(self.save_path, pkl_name), 'wb') as handle:
#             pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         #
        
#     def save_model_state(self, model):
        
#         state_name = 'play{}.dt'.format(len(self.hist_rewards))
#         torch.save(model.state_dict(), os.path.join(self.save_path, state_name))
        
#     def update(self, s, a, r):
        
#         state = s.reshape([1,-1])
#         action = a.reshape([1,-1])
#         reward = r.reshape([1,-1])
        
#         state = self.game.env.scaler_S.inverse_transform(state)
#         reward = np.power(10, -r)
        
#         self.hist_states.append(state)
#         self.hist_actions.append(action)
#         self.hist_rewards.append(reward)
        
#         self.V += np.squeeze(reward)
        
#         nplay = len(self.hist_states)
        
#         if self.logger is not None:    
#             self.logger.info('=========================================')
#             self.logger.info('             Play {} '.format(nplay))
#             self.logger.info('=========================================') 
            
#             self.logger.info('\n### state  = {} ###'.format(state))
#             self.logger.info('\n### action = {} ###'.format(action))
#             self.logger.info('\n### reward = {} ###'.format(reward))
#             self.logger.info('\n### V      = {} ###'.format(self.V))

#         #
        
#         self._dump_meter()


# class BanditTrunkMeters(object):
    
#     def __init__(self, game, save_path, logger):
    
#         self.save_path = save_path
#         self.logger = logger
#         self.game = game
        
#         self.hist_states = []
#         self.hist_action_series = []
#         self.hist_trunk_rewards = []
#         self.V = np.zeros(game.env.n_trunks)

#     def _dump_meter(self,):
        
#         res = {}
        
#         res['hist_states'] = np.vstack(self.hist_states)
#         res['hist_action_series'] = np.vstack(self.hist_action_series)
#         res['hist_trunk_rewards'] = np.vstack(self.hist_trunk_rewards)
#         res['V'] = self.V
        
#         pkl_name = 'play{}.pickle'.format(len(self.hist_states))

#         with open(os.path.join(self.save_path, pkl_name), 'wb') as handle:
#             pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         #
        
#     def save_model_state(self, models):
        
#         for i in range(len(models)):
#             state_name = 'play{}_part{}.dt'.format(len(self.hist_states), i)
#             torch.save(models[i].state_dict(), os.path.join(self.save_path, state_name))
        
#     def update(self, state, action_series, trunk_rewards):
        
#         state = state.reshape([1,-1])
#         state = self.game.env.scaler_S.inverse_transform(state)
        
#         action_series = np.expand_dims(action_series, 0)
        
#         trunk_rewards = trunk_rewards.reshape([1,-1])
#         trunk_rewards = np.power(10, -trunk_rewards)
        
#         self.V += np.squeeze(trunk_rewards)
        
#         self.hist_states.append(state)
#         self.hist_action_series.append(action_series)
#         self.hist_trunk_rewards.append(trunk_rewards)
        
# #         print(np.vstack(self.hist_states).shape)
# #         print(np.vstack(self.hist_action_series).shape)
# #         print(np.vstack(self.hist_trunk_rewards).shape)
        
#         nplay = len(self.hist_states)
        
#         if self.logger is not None:    
#             self.logger.info('=========================================')
#             self.logger.info('             Play {} '.format(nplay))
#             self.logger.info('=========================================') 
            
#             self.logger.info('\n### state  = \n{} '.format(state))
#             self.logger.info('\n### action = \n{} '.format(action_series))
#             self.logger.info('\n### reward = \n{} '.format(trunk_rewards))
#             self.logger.info('\n### V      = \n{} '.format(self.V))

#         #
        
#         self._dump_meter()
        

        
