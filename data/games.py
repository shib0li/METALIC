import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import itertools

from infras.randutils import *
from infras.misc import *

class GamePoisson:
    
    def __init__(self, xpinn_client=None):
        
        self.dim_S = 1
        self.dim_A = 9
        
        self.lb_S = np.array([1])
        self.ub_S = np.array([50])
        
        self._init_context_scalers()
        
        if xpinn_client is None:
            raise Exception('Error, no XPINN client is given')
        else:
            self.xpinn_client = xpinn_client
            
#         self.opts = self.xpinn_client.num_adam + \
#                 self.xpinn_client.num_lbfgs

        self.opts = self.xpinn_client.opts

    def _init_actions(self,):
        Alist = [list(i) for i in itertools.product([0, 1], repeat=self.dim_A)]
        A = np.array(Alist)
        return A
    
    def _init_context_scalers(self, ):
        S = generate_with_bounds(100, self.lb_S, self.ub_S, 'linspace')
        self.scaler_S = StandardScaler()
        self.scaler_S.fit(S)
 
    def sample_states(self, n_sample, seed):
        S = generate_with_bounds(n_sample, self.lb_S, self.ub_S, 'uniform', seed)
        S = self.scaler_S.transform(S)
        return S
    
    def sample_actions(self,):
        A = self._init_actions()
        return A

    def reward_query(self, state, action):

        if state.ndim == 1:
            state = state.reshape([1,-1])
        if action.ndim == 1:
            action = action.reshape([1,-1])
            
        assert state.shape[0] == action.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
        
        sharpness = state[0,0]
        self.xpinn_client.reset(sharpness)
        
        errors = self.xpinn_client.query(interface=action.squeeze())
        #errors = np.random.uniform(0, 20, size=action.shape[1])
        
        r = -np.log10(errors[-1]).astype(float)
        
        return r
    
    def reward_batch(self, state, actions):

        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        #assert actions.shape[0] == self.xpinn_client.num_adam+self.xpinn_client.num_lbfgs
        assert actions.shape[0] == self.opts
        
        state = self.scaler_S.inverse_transform(state)
            
        sharpness = state[0,0]
        self.xpinn_client.reset(sharpness)
        
        errors = self.xpinn_client.batch(interface=actions)
        #errors = np.random.uniform(0, 20, size=actions.shape[0])
        
        r = -np.log10(errors).astype(float)
        
        return r
    
    
    def reward_step(self, state, action):
        
        if state.ndim == 1:
            state = state.reshape([1,-1])
        if action.ndim == 1:
            action = action.reshape([1,-1])
            
        assert state.shape[0] == action.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
            
        sharpness = state[0,0]
        
        if np.abs(self.xpinn_client.pde_param-sharpness) > 1e-5:
            # switch context
            self.xpinn_client.reset(sharpness)

        err = self.xpinn_client.step(interface=action.squeeze())
        #err = np.random.uniform(0, 20)
        
        r = -np.log10(err).astype(float)
        
        return r
    
class GameHeat:
    
    def __init__(self, xpinn_client=None):
        
        self.dim_S = 4
        self.dim_A = 9
        
        self.lb_S = np.array([-2.0, -2.0, -2.0, -2.0])
        self.ub_S = np.array([2.0, 2.0, 2.0, 2.0])
        
        self._init_context_scalers()
        
        if xpinn_client is None:
            raise Exception('Error, no XPINN client is given')
        else:
            self.xpinn_client = xpinn_client

        self.opts = self.xpinn_client.opts

    def _init_actions(self,):
        Alist = [list(i) for i in itertools.product([0, 1], repeat=self.dim_A)]
        A = np.array(Alist)
        return A
    
    def _init_context_scalers(self, ):
        #S = generate_with_bounds(100, self.lb_S, self.ub_S, 'linspace')
        S = generate_with_bounds(5000, self.lb_S, self.ub_S, 'lhs', seed=0)
        self.scaler_S = StandardScaler()
        self.scaler_S.fit(S)
 
    def sample_states(self, n_sample, seed):
        S = generate_with_bounds(n_sample, self.lb_S, self.ub_S, 'uniform', seed)
        S = self.scaler_S.transform(S)
        return S
    
    def sample_actions(self,):
        A = self._init_actions()
        return A

    def reward_query(self, state, action):

        if state.ndim == 1:
            state = state.reshape([1,-1])
        if action.ndim == 1:
            action = action.reshape([1,-1])
            
        assert state.shape[0] == action.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
        
        param = state.squeeze()
        self.xpinn_client.reset(param)
        
        errors = self.xpinn_client.query(interface=action.squeeze())
        #errors = np.random.uniform(0, 20, size=action.shape[1])
        
        r = -np.log10(errors[-1]).astype(float)
        
        return r
    
    def reward_batch(self, state, actions):

        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        #assert actions.shape[0] == self.xpinn_client.num_adam+self.xpinn_client.num_lbfgs
        assert actions.shape[0] == self.opts
        
        state = self.scaler_S.inverse_transform(state)
            
        param = state.squeeze()
        self.xpinn_client.reset(param)
        
        errors = self.xpinn_client.batch(interface=actions)
        #errors = np.random.uniform(0, 20, size=actions.shape[0])
        
        r = -np.log10(errors).astype(float)
        
        return r
    
    
    def reward_step(self, state, action):
        
        if state.ndim == 1:
            state = state.reshape([1,-1])
        if action.ndim == 1:
            action = action.reshape([1,-1])
            
        assert state.shape[0] == action.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
            
        param = state.squeeze()
        
        if np.sum(np.abs(self.xpinn_client.pde_param-param)) > 1e-5:
            # switch context
            self.xpinn_client.reset(param)

        err = self.xpinn_client.step(interface=action.squeeze())
        #err = np.random.uniform(0, 20)
        
        r = -np.log10(err).astype(float)
        
        return r
    
    
class GameBurgers:
    
    def __init__(self, xpinn_client=None):
        
        self.dim_S = 1
        self.dim_A = 9
        
#         self.lb_S = np.array([0.0001])
#         self.ub_S = np.array([0.01])

        self.lb_S = np.array([0.001])
        self.ub_S = np.array([0.05])
        
        self._init_context_scalers()
        
        if xpinn_client is None:
            raise Exception('Error, no XPINN client is given')
        else:
            self.xpinn_client = xpinn_client

        self.opts = self.xpinn_client.opts

    def _init_actions(self,):
        Alist = [list(i) for i in itertools.product([0, 1], repeat=self.dim_A)]
        A = np.array(Alist)
        return A
    
    def _init_context_scalers(self, ):
        S = generate_with_bounds(100, self.lb_S, self.ub_S, 'linspace')
        self.scaler_S = StandardScaler()
        self.scaler_S.fit(S)
 
    def sample_states(self, n_sample, seed):
        S = generate_with_bounds(n_sample, self.lb_S, self.ub_S, 'uniform', seed)
        S = self.scaler_S.transform(S)
        return S
    
    def sample_actions(self,):
        A = self._init_actions()
        return A

    def reward_query(self, state, action):

        if state.ndim == 1:
            state = state.reshape([1,-1])
        if action.ndim == 1:
            action = action.reshape([1,-1])
            
        assert state.shape[0] == action.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
        
        param = state[0,0]
        self.xpinn_client.reset(param)
        
        errors = self.xpinn_client.query(interface=action.squeeze())
        #errors = np.random.uniform(0, 20, size=action.shape[1])
        
        r = -np.log10(errors[-1]).astype(float)
        
        return r
    
    def reward_batch(self, state, actions):

        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        #assert actions.shape[0] == self.xpinn_client.num_adam+self.xpinn_client.num_lbfgs
        assert actions.shape[0] == self.opts
        
        state = self.scaler_S.inverse_transform(state)
            
        param = state[0,0]
        self.xpinn_client.reset(param)
        
        errors = self.xpinn_client.batch(interface=actions)
        #errors = np.random.uniform(0, 20, size=actions.shape[0])
        
        r = -np.log10(errors).astype(float)
        
        return r
    
    
    def reward_step(self, state, action):
        
        if state.ndim == 1:
            state = state.reshape([1,-1])
        if action.ndim == 1:
            action = action.reshape([1,-1])
            
        assert state.shape[0] == action.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
            
        param = state[0,0]
        
        if np.abs(self.xpinn_client.pde_param-param) > 1e-5:
            # switch context
            self.xpinn_client.reset(param)

        err = self.xpinn_client.step(interface=action.squeeze())
        #err = np.random.uniform(0, 20)
        
        r = -np.log10(err).astype(float)
        
        return r
    
    
class GameKdV:
    
    def __init__(self, xpinn_client=None):
        
        self.dim_S = 1
        self.dim_A = 9
        
        self.lb_S = np.array([1.0])
        self.ub_S = np.array([10.0])
        
        self._init_context_scalers()
        
        if xpinn_client is None:
            raise Exception('Error, no XPINN client is given')
        else:
            self.xpinn_client = xpinn_client

        self.opts = self.xpinn_client.opts

    def _init_actions(self,):
        Alist = [list(i) for i in itertools.product([0, 1], repeat=self.dim_A)]
        A = np.array(Alist)
        return A
    
    def _init_context_scalers(self, ):
        S = generate_with_bounds(100, self.lb_S, self.ub_S, 'linspace')
        self.scaler_S = StandardScaler()
        self.scaler_S.fit(S)
 
    def sample_states(self, n_sample, seed):
        S = generate_with_bounds(n_sample, self.lb_S, self.ub_S, 'uniform', seed)
        S = self.scaler_S.transform(S)
        return S
    
    def sample_actions(self,):
        A = self._init_actions()
        return A

    def reward_query(self, state, action):

        if state.ndim == 1:
            state = state.reshape([1,-1])
        if action.ndim == 1:
            action = action.reshape([1,-1])
            
        assert state.shape[0] == action.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
        
        param = state[0,0]
        self.xpinn_client.reset(param)
        
        errors = self.xpinn_client.query(interface=action.squeeze())
        #errors = np.random.uniform(0, 20, size=action.shape[1])
        
        r = -np.log10(errors[-1]).astype(float)
        
        return r
    
    def reward_batch(self, state, actions):

        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        #assert actions.shape[0] == self.xpinn_client.num_adam+self.xpinn_client.num_lbfgs
        assert actions.shape[0] == self.opts
        
        state = self.scaler_S.inverse_transform(state)
            
        param = state[0,0]
        self.xpinn_client.reset(param)
        
        errors = self.xpinn_client.batch(interface=actions)
        #errors = np.random.uniform(0, 20, size=actions.shape[0])
        
        r = -np.log10(errors).astype(float)
        
        return r
    
    
    def reward_step(self, state, action):
        
        if state.ndim == 1:
            state = state.reshape([1,-1])
        if action.ndim == 1:
            action = action.reshape([1,-1])
            
        assert state.shape[0] == action.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
            
        param = state[0,0]
        
        if np.abs(self.xpinn_client.pde_param-param) > 1e-5:
            # switch context
            self.xpinn_client.reset(param)

        err = self.xpinn_client.step(interface=action.squeeze())
        #err = np.random.uniform(0, 20)
        
        r = -np.log10(err).astype(float)
        
        return r
    
class GameAdvec:
    
    def __init__(self, xpinn_client=None):
        
        self.dim_S = 1
        self.dim_A = 9
        
        self.lb_S = np.array([1.0])
        self.ub_S = np.array([30.0])
        
        self._init_context_scalers()
        
        if xpinn_client is None:
            raise Exception('Error, no XPINN client is given')
        else:
            self.xpinn_client = xpinn_client

        self.opts = self.xpinn_client.opts

    def _init_actions(self,):
        Alist = [list(i) for i in itertools.product([0, 1], repeat=self.dim_A)]
        A = np.array(Alist)
        return A
    
    def _init_context_scalers(self, ):
        S = generate_with_bounds(100, self.lb_S, self.ub_S, 'linspace')
        self.scaler_S = StandardScaler()
        self.scaler_S.fit(S)
 
    def sample_states(self, n_sample, seed):
        S = generate_with_bounds(n_sample, self.lb_S, self.ub_S, 'uniform', seed)
        S = self.scaler_S.transform(S)
        return S
    
    def sample_actions(self,):
        A = self._init_actions()
        return A

    def reward_query(self, state, action):

        if state.ndim == 1:
            state = state.reshape([1,-1])
        if action.ndim == 1:
            action = action.reshape([1,-1])
            
        assert state.shape[0] == action.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
        
        param = state[0,0]
        self.xpinn_client.reset(param)
        
        errors = self.xpinn_client.query(interface=action.squeeze())
        #errors = np.random.uniform(0, 20, size=action.shape[1])
        
        r = -np.log10(errors[-1]).astype(float)
        
        return r
    
    def reward_batch(self, state, actions):

        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        #assert actions.shape[0] == self.xpinn_client.num_adam+self.xpinn_client.num_lbfgs
        assert actions.shape[0] == self.opts
        
        state = self.scaler_S.inverse_transform(state)
            
        param = state[0,0]
        self.xpinn_client.reset(param)
        
        errors = self.xpinn_client.batch(interface=actions)
        #errors = np.random.uniform(0, 20, size=actions.shape[0])
        
        r = -np.log10(errors).astype(float)
        
        return r
    
    
    def reward_step(self, state, action):
        
        if state.ndim == 1:
            state = state.reshape([1,-1])
        if action.ndim == 1:
            action = action.reshape([1,-1])
            
        assert state.shape[0] == action.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
            
        param = state[0,0]
        
        if np.abs(self.xpinn_client.pde_param-param) > 1e-5:
            # switch context
            self.xpinn_client.reset(param)

        err = self.xpinn_client.step(interface=action.squeeze())
        #err = np.random.uniform(0, 20)
        
        r = -np.log10(err).astype(float)
        
        return r
    
class GameReaction:
    
    def __init__(self, xpinn_client=None):
        
        self.dim_S = 1
        self.dim_A = 9
        
        self.lb_S = np.array([1.0])
        self.ub_S = np.array([15.0])
        
        self._init_context_scalers()
        
        if xpinn_client is None:
            raise Exception('Error, no XPINN client is given')
        else:
            self.xpinn_client = xpinn_client

        self.opts = self.xpinn_client.opts

    def _init_actions(self,):
        Alist = [list(i) for i in itertools.product([0, 1], repeat=self.dim_A)]
        A = np.array(Alist)
        return A
    
    def _init_context_scalers(self, ):
        S = generate_with_bounds(100, self.lb_S, self.ub_S, 'linspace')
        self.scaler_S = StandardScaler()
        self.scaler_S.fit(S)
 
    def sample_states(self, n_sample, seed):
        S = generate_with_bounds(n_sample, self.lb_S, self.ub_S, 'uniform', seed)
        S = self.scaler_S.transform(S)
        return S
    
    def sample_actions(self,):
        A = self._init_actions()
        return A

    def reward_query(self, state, action):

        if state.ndim == 1:
            state = state.reshape([1,-1])
        if action.ndim == 1:
            action = action.reshape([1,-1])
            
        assert state.shape[0] == action.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
        
        param = state[0,0]
        self.xpinn_client.reset(param)
        
        errors = self.xpinn_client.query(interface=action.squeeze())
        #errors = np.random.uniform(0, 20, size=action.shape[1])
        
        r = -np.log10(errors[-1]).astype(float)
        
        return r
    
    def reward_batch(self, state, actions):

        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        #assert actions.shape[0] == self.xpinn_client.num_adam+self.xpinn_client.num_lbfgs
        assert actions.shape[0] == self.opts
        
        state = self.scaler_S.inverse_transform(state)
            
        param = state[0,0]
        self.xpinn_client.reset(param)
        
        errors = self.xpinn_client.batch(interface=actions)
        #errors = np.random.uniform(0, 20, size=actions.shape[0])
        
        r = -np.log10(errors).astype(float)
        
        return r
    
    
    def reward_step(self, state, action):
        
        if state.ndim == 1:
            state = state.reshape([1,-1])
        if action.ndim == 1:
            action = action.reshape([1,-1])
            
        assert state.shape[0] == action.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
            
        param = state[0,0]
        
        if np.abs(self.xpinn_client.pde_param-param) > 1e-5:
            # switch context
            self.xpinn_client.reset(param)

        err = self.xpinn_client.step(interface=action.squeeze())
        #err = np.random.uniform(0, 20)
        
        r = -np.log10(err).astype(float)
        
        return r
    
    
class PINNPoisson:
    
    def __init__(self, pinn_client=None):
        
        self.dim_S = 1
        self.dim_A = 9
        
        self.lb_S = np.array([1])
        self.ub_S = np.array([50])
        
        self._init_context_scalers()
        
        if pinn_client is None:
            raise Exception('Error, no XPINN client is given')
        else:
            self.pinn_client = pinn_client
            
        self.opts = self.pinn_client.num_adam + \
                self.pinn_client.num_lbfgs

    def _init_actions(self,):
        Alist = [list(i) for i in itertools.product([0, 1], repeat=self.dim_A)]
        A = np.array(Alist)
        return A
    
    def _init_context_scalers(self, ):
        S = generate_with_bounds(100, self.lb_S, self.ub_S, 'linspace')
        self.scaler_S = StandardScaler()
        self.scaler_S.fit(S)
 
    def sample_states(self, n_sample, seed):
        S = generate_with_bounds(n_sample, self.lb_S, self.ub_S, 'uniform', seed)
        S = self.scaler_S.transform(S)
        return S
    
    def sample_actions(self,):
        A = self._init_actions()
        return A

    def reward_query(self, state):

        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        assert state.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
        
        sharpness = state[0,0]
        self.pinn_client.reset(sharpness)
        
        errors = self.pinn_client.query()
        #errors = np.random.uniform(0, 20, size=action.shape[1])
        
        r = -np.log10(errors[-1]).astype(float)
        
        return r
    
    
    def reward_step(self, state):
        
        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        assert state.shape[0]  == 1
        
        state = self.scaler_S.inverse_transform(state)
            
        sharpness = state[0,0]
        
        if np.abs(self.xpinn_client.pde_param-sharpness) > 1e-5:
            # switch context
            self.xpinn_client.reset(sharpness)

        err = self.pinn_client.step()
        #err = np.random.uniform(0, 20)
        
        r = -np.log10(err).astype(float)
        
        return r
    
class PINNHeat:
    
    def __init__(self, pinn_client=None):
        
        self.dim_S = 4
        self.dim_A = 9
        
        self.lb_S = np.array([-2.0, -2.0, -2.0, -2.0])
        self.ub_S = np.array([2.0, 2.0, 2.0, 2.0])
        
        self._init_context_scalers()
        
        if pinn_client is None:
            raise Exception('Error, no XPINN client is given')
        else:
            self.pinn_client = pinn_client

        self.opts = self.pinn_client.num_adam + \
                self.pinn_client.num_lbfgs

    def _init_actions(self,):
        Alist = [list(i) for i in itertools.product([0, 1], repeat=self.dim_A)]
        A = np.array(Alist)
        return A
    
    def _init_context_scalers(self, ):
        #S = generate_with_bounds(100, self.lb_S, self.ub_S, 'linspace')
        S = generate_with_bounds(5000, self.lb_S, self.ub_S, 'lhs', seed=0)
        self.scaler_S = StandardScaler()
        self.scaler_S.fit(S)
 
    def sample_states(self, n_sample, seed):
        S = generate_with_bounds(n_sample, self.lb_S, self.ub_S, 'uniform', seed)
        S = self.scaler_S.transform(S)
        return S
    
    def sample_actions(self,):
        A = self._init_actions()
        return A

    def reward_query(self, state):

        if state.ndim == 1:
            state = state.reshape([1,-1])
#         if action.ndim == 1:
#             action = action.reshape([1,-1])
            
        assert state.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
        
        param = state.squeeze()
        self.pinn_client.reset(param)
        
        errors = self.pinn_client.query()
        
        r = -np.log10(errors[-1]).astype(float)
        
        return r
    
class PINNBurgers:
    
    def __init__(self, pinn_client=None):
        
        self.dim_S = 1
        self.dim_A = 9
        
#         self.lb_S = np.array([0.0001])
#         self.ub_S = np.array([0.01])

        self.lb_S = np.array([0.001])
        self.ub_S = np.array([0.05])
        
        self._init_context_scalers()
        
        if pinn_client is None:
            raise Exception('Error, no XPINN client is given')
        else:
            self.pinn_client = pinn_client
            
        self.opts = self.pinn_client.num_adam + \
                self.pinn_client.num_lbfgs

    def _init_actions(self,):
        Alist = [list(i) for i in itertools.product([0, 1], repeat=self.dim_A)]
        A = np.array(Alist)
        return A
    
    def _init_context_scalers(self, ):
        S = generate_with_bounds(100, self.lb_S, self.ub_S, 'linspace')
        self.scaler_S = StandardScaler()
        self.scaler_S.fit(S)
 
    def sample_states(self, n_sample, seed):
        S = generate_with_bounds(n_sample, self.lb_S, self.ub_S, 'uniform', seed)
        S = self.scaler_S.transform(S)
        return S
    
    def sample_actions(self,):
        A = self._init_actions()
        return A

    def reward_query(self, state):

        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        assert state.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
        
        param = state[0,0]
        self.pinn_client.reset(param)
        
        errors = self.pinn_client.query()
        #errors = np.random.uniform(0, 20, size=action.shape[1])
        
        r = -np.log10(errors[-1]).astype(float)
        
        return r
    
    
    def reward_step(self, state):
        
        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        assert state.shape[0]  == 1
        
        state = self.scaler_S.inverse_transform(state)
            
        param = state[0,0]
        
        if np.abs(self.xpinn_client.pde_param-param) > 1e-5:
            # switch context
            self.xpinn_client.reset(param)

        err = self.pinn_client.step()
        #err = np.random.uniform(0, 20)
        
        r = -np.log10(err).astype(float)
        
        return r
    
class PINNKdV:
    
    def __init__(self, pinn_client=None):
        
        self.dim_S = 1
        self.dim_A = 9
        
        self.lb_S = np.array([1.0])
        self.ub_S = np.array([10.0])
        
        self._init_context_scalers()
        
        if pinn_client is None:
            raise Exception('Error, no XPINN client is given')
        else:
            self.pinn_client = pinn_client
            
        self.opts = self.pinn_client.num_adam + \
                self.pinn_client.num_lbfgs

    def _init_actions(self,):
        Alist = [list(i) for i in itertools.product([0, 1], repeat=self.dim_A)]
        A = np.array(Alist)
        return A
    
    def _init_context_scalers(self, ):
        S = generate_with_bounds(100, self.lb_S, self.ub_S, 'linspace')
        self.scaler_S = StandardScaler()
        self.scaler_S.fit(S)
 
    def sample_states(self, n_sample, seed):
        S = generate_with_bounds(n_sample, self.lb_S, self.ub_S, 'uniform', seed)
        S = self.scaler_S.transform(S)
        return S
    
    def sample_actions(self,):
        A = self._init_actions()
        return A

    def reward_query(self, state):

        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        assert state.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
        
        param = state[0,0]
        self.pinn_client.reset(param)
        
        errors = self.pinn_client.query()
        #errors = np.random.uniform(0, 20, size=action.shape[1])
        
        r = -np.log10(errors[-1]).astype(float)
        
        return r
    
    
    def reward_step(self, state):
        
        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        assert state.shape[0]  == 1
        
        state = self.scaler_S.inverse_transform(state)
            
        param = state[0,0]
        
        if np.abs(self.xpinn_client.pde_param-param) > 1e-5:
            # switch context
            self.xpinn_client.reset(param)

        err = self.pinn_client.step()
        #err = np.random.uniform(0, 20)
        
        r = -np.log10(err).astype(float)
        
        return r
    
class PINNKdV:
    
    def __init__(self, pinn_client=None):
        
        self.dim_S = 1
        self.dim_A = 9
        
        self.lb_S = np.array([1.0])
        self.ub_S = np.array([10.0])
        
        self._init_context_scalers()
        
        if pinn_client is None:
            raise Exception('Error, no XPINN client is given')
        else:
            self.pinn_client = pinn_client
            
        self.opts = self.pinn_client.num_adam + \
                self.pinn_client.num_lbfgs

    def _init_actions(self,):
        Alist = [list(i) for i in itertools.product([0, 1], repeat=self.dim_A)]
        A = np.array(Alist)
        return A
    
    def _init_context_scalers(self, ):
        S = generate_with_bounds(100, self.lb_S, self.ub_S, 'linspace')
        self.scaler_S = StandardScaler()
        self.scaler_S.fit(S)
 
    def sample_states(self, n_sample, seed):
        S = generate_with_bounds(n_sample, self.lb_S, self.ub_S, 'uniform', seed)
        S = self.scaler_S.transform(S)
        return S
    
    def sample_actions(self,):
        A = self._init_actions()
        return A

    def reward_query(self, state):

        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        assert state.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
        
        param = state[0,0]
        self.pinn_client.reset(param)
        
        errors = self.pinn_client.query()
        #errors = np.random.uniform(0, 20, size=action.shape[1])
        
        r = -np.log10(errors[-1]).astype(float)
        
        return r
    
    
    def reward_step(self, state):
        
        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        assert state.shape[0]  == 1
        
        state = self.scaler_S.inverse_transform(state)
            
        param = state[0,0]
        
        if np.abs(self.xpinn_client.pde_param-param) > 1e-5:
            # switch context
            self.xpinn_client.reset(param)

        err = self.pinn_client.step()
        #err = np.random.uniform(0, 20)
        
        r = -np.log10(err).astype(float)
        
        return r
    
    
class PINNAdvec:
    
    def __init__(self, pinn_client=None):
        
        self.dim_S = 1
        self.dim_A = 9
        
        self.lb_S = np.array([1.0])
        self.ub_S = np.array([30.0])
        
        self._init_context_scalers()
        
        if pinn_client is None:
            raise Exception('Error, no XPINN client is given')
        else:
            self.pinn_client = pinn_client
            
        self.opts = self.pinn_client.num_adam + \
                self.pinn_client.num_lbfgs

    def _init_actions(self,):
        Alist = [list(i) for i in itertools.product([0, 1], repeat=self.dim_A)]
        A = np.array(Alist)
        return A
    
    def _init_context_scalers(self, ):
        S = generate_with_bounds(100, self.lb_S, self.ub_S, 'linspace')
        self.scaler_S = StandardScaler()
        self.scaler_S.fit(S)
 
    def sample_states(self, n_sample, seed):
        S = generate_with_bounds(n_sample, self.lb_S, self.ub_S, 'uniform', seed)
        S = self.scaler_S.transform(S)
        return S
    
    def sample_actions(self,):
        A = self._init_actions()
        return A

    def reward_query(self, state):

        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        assert state.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
        
        param = state[0,0]
        self.pinn_client.reset(param)
        
        errors = self.pinn_client.query()
        #errors = np.random.uniform(0, 20, size=action.shape[1])
        
        r = -np.log10(errors[-1]).astype(float)
        
        return r
    
    
    def reward_step(self, state):
        
        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        assert state.shape[0]  == 1
        
        state = self.scaler_S.inverse_transform(state)
            
        param = state[0,0]
        
        if np.abs(self.xpinn_client.pde_param-param) > 1e-5:
            # switch context
            self.xpinn_client.reset(param)

        err = self.pinn_client.step()
        #err = np.random.uniform(0, 20)
        
        r = -np.log10(err).astype(float)
        
        return r
    
class PINNReaction:
    
    def __init__(self, pinn_client=None):
        
        self.dim_S = 1
        self.dim_A = 9
        
        self.lb_S = np.array([1.0])
        self.ub_S = np.array([15.0])
        
        self._init_context_scalers()
        
        if pinn_client is None:
            raise Exception('Error, no XPINN client is given')
        else:
            self.pinn_client = pinn_client
            
        self.opts = self.pinn_client.num_adam + \
                self.pinn_client.num_lbfgs

    def _init_actions(self,):
        Alist = [list(i) for i in itertools.product([0, 1], repeat=self.dim_A)]
        A = np.array(Alist)
        return A
    
    def _init_context_scalers(self, ):
        S = generate_with_bounds(100, self.lb_S, self.ub_S, 'linspace')
        self.scaler_S = StandardScaler()
        self.scaler_S.fit(S)
 
    def sample_states(self, n_sample, seed):
        S = generate_with_bounds(n_sample, self.lb_S, self.ub_S, 'uniform', seed)
        S = self.scaler_S.transform(S)
        return S
    
    def sample_actions(self,):
        A = self._init_actions()
        return A

    def reward_query(self, state):

        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        assert state.shape[0] == 1
        
        state = self.scaler_S.inverse_transform(state)
        
        param = state[0,0]
        self.pinn_client.reset(param)
        
        errors = self.pinn_client.query()
        #errors = np.random.uniform(0, 20, size=action.shape[1])
        
        r = -np.log10(errors[-1]).astype(float)
        
        return r
    
    
    def reward_step(self, state):
        
        if state.ndim == 1:
            state = state.reshape([1,-1])
            
        assert state.shape[0]  == 1
        
        state = self.scaler_S.inverse_transform(state)
            
        param = state[0,0]
        
        if np.abs(self.xpinn_client.pde_param-param) > 1e-5:
            # switch context
            self.xpinn_client.reset(param)

        err = self.pinn_client.step()
        #err = np.random.uniform(0, 20)
        
        r = -np.log10(err).astype(float)
        
        return r
