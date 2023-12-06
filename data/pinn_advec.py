import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.tri as tri
import time
import sympy as sy
import copy
# from pyDOE import lhs
# from scipy.special import erf

from scipy.integrate import odeint

import warnings
from infras.randutils import *
from infras.misc import *


# def advec_solution(k, beta):
    
#     N = 256
#     M = 101
#     x = np.linspace(0., 2*np.pi, N)
#     t = np.linspace(0., 1.0, M) 
#     X, T = np.meshgrid(x, t)
#     sol = np.sin(k*(X.flatten() - beta* T.flatten()))
#     sol = sol.reshape([M, N])
#     return (x,t,sol)


def advec_solver(x, t, param):
    
    if x.ndim == 1:
        x = x.reshape([-1,1])
        
    if t.ndim == 1:
        t = t.reshape([-1,1])
        
    k=1.0
    beta=param
    
    u = np.sin(k*(x - beta* t))
    
    return u



def gen_advec_data(
    param,
    Nu=100,
    Nf=1000,
):

    x_max=2*np.pi
    t_max=1.0

    t_split = 0.5

    x_lb=0.0
    x_ub=2*np.pi
    
    t_lb=0.0
    t_ub=1.0
    
    x_grid=256
    t_grid=101
    
    t_split = 0.5
    
    x = np.linspace(x_lb, x_ub, x_grid)
    t = np.linspace(t_lb, t_ub, t_grid)
    
    #print(x)
    #print(t)
    
    x_left = x_lb*np.ones([x_grid,1])
    x_right = x_ub*np.ones([x_grid,1])
    
    t_left = t_lb*np.ones([t_grid,1])
    t_right = t_ub*np.ones([t_grid,1])
    
    #print(x_left)
    #print(x_right)
    #print(t_left)
    #print(t_right)
    

    bounds = np.vstack((
        np.hstack((x_left, np.linspace(t_lb, t_ub, x_grid).reshape([-1,1]))),
        np.hstack((x_right, np.linspace(t_lb, t_ub, x_grid).reshape([-1,1]))),
        np.hstack((np.linspace(x_lb, x_ub, t_grid).reshape([-1,1]), t_left)),
        np.hstack((np.linspace(x_lb, x_ub, t_grid).reshape([-1,1]), t_right))
    ))
    
    bound1 = bounds[bounds[:,1] < t_split]
    bound2 = bounds[bounds[:,1] > t_split]
    
    #print(bound1.shape)
    #print(bound2.shape)
    
    idx_bound1 = generate_random_choice(a=bound1.shape[0], N=Nu, seed=1)
    idx_bound2 = generate_random_choice(a=bound2.shape[0], N=Nu, seed=2)
    
    bound1 = bound1[idx_bound1,:]
    bound2 = bound2[idx_bound2,:]
    
    #print(bound1.shape)
    #print(bound2.shape)

    margin = 0.01
    
    lb1 = np.array([x_lb+margin, t_lb+margin])
    ub1 = np.array([x_ub-margin, t_split-margin])
    
    lb2 = np.array([x_lb+margin, t_split+margin])
    ub2 = np.array([x_ub-margin, t_ub-margin])

    colls1 = generate_with_bounds(N=1000, lb=lb1, ub=ub1, method='lhs', seed=3)
    colls2 = generate_with_bounds(N=1000, lb=lb2, ub=ub2, method='lhs', seed=4)
    
    num_i = 101
    
    interface = np.hstack((
        np.linspace(x_lb, x_ub, num_i).reshape([-1,1]),
        t_split*np.ones([num_i,1])
    ))
    
    #print(interface.shape)
    #print(interface)
    
    mesh_x, mesh_t = np.meshgrid(x, t)
    
    S = np.vstack((mesh_x.flatten(), mesh_t.flatten())).T
    U = advec_solver(mesh_x.flatten(), mesh_t.flatten(), param)
    #print(U.shape)
    
    
    S1 = S[S[:,1]<t_split]
    S2 = S[S[:,1]>t_split]
    
    
    #print(S1.shape)
    #print(S2.shape)
    
    task_data = {}

    task_data['xb1'] = bound1[:,0]
    task_data['yb1'] = bound1[:,1]
    task_data['ub1'] = advec_solver(bound1[:,0], bound1[:,1], param).squeeze()

    task_data['xb2'] = bound2[:,0]
    task_data['yb2'] = bound2[:,1]
    task_data['ub2'] = advec_solver(bound2[:,0], bound2[:,1], param).squeeze()

    task_data['xf1'] = colls1[:,0]
    task_data['yf1'] = colls1[:,1]
    task_data['ff1'] = advec_solver(colls1[:,0], colls1[:,1], param).squeeze()

    task_data['xf2'] = colls2[:,0]
    task_data['yf2'] = colls2[:,1]
    task_data['ff2'] = advec_solver(colls2[:,0], colls2[:,1], param).squeeze()

    task_data['xi1'] = interface[:,0]
    task_data['yi1'] = interface[:,1]
    task_data['fi1'] = advec_solver(interface[:,0], interface[:,1], param).squeeze()

    task_data['x1'] = S1[:,0]
    task_data['y1'] = S1[:,1]
    task_data['u1'] = advec_solver(S1[:,0], S1[:,1], param).squeeze()

    task_data['x2'] = S2[:,0]
    task_data['y2'] = S2[:,1]
    task_data['u2'] = advec_solver(S2[:,0], S2[:,1], param).squeeze()
    
    task_data['param'] = param
    
    task_data['x_span'] = mesh_x.flatten()
    task_data['t_span'] = mesh_t.flatten()
    task_data['u_span'] = U.flatten()

    task_data['xlb1'] = S1[:,0].min()
    task_data['xub1'] = S1[:,0].max()
    
    task_data['xlb2'] = S2[:,0].min()
    task_data['xub2'] = S2[:,0].max()
    
    task_data['ylb1'] = S1[:,1].min()
    task_data['yub1'] = S1[:,1].max()
    
    task_data['ylb2'] = S2[:,1].min()
    task_data['yub2'] = S2[:,1].max()
    
    return task_data
           


# param = np.array([0.4,0.4, 0.9])

# data = gen_heat_data(param)

# fig=plt.figure(figsize=(12,8), dpi= 100, facecolor='w', edgecolor='k')

# plt.scatter(data['xb1'], data['yb1'], color='b')
# plt.scatter(data['xb2'], data['yb2'], color='k')
# plt.scatter(data['xf1'], data['yf1'])
# plt.scatter(data['xf2'], data['yf2'])
# plt.scatter(data['xi1'], data['yi1'], color='g')

    
class Net(nn.Module):
    def __init__(self, layers, lb, ub, act=nn.Tanh()):
        super(Net, self).__init__()
        self.act = act
        self.fc = nn.ModuleList()
#         self.lb = torch.tensor(lb, dtype=torch.float32)
#         self.ub = torch.tensor(ub, dtype=torch.float32)
        self.register_buffer('lb', torch.tensor(lb).to(torch.float32))
        self.register_buffer('ub', torch.tensor(ub).to(torch.float32))
        for i in range(len(layers) - 1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_normal_(self.fc[-1].weight)
        
    def forward(self, x):
        x = 2.0*(x - self.lb)/(self.ub - self.lb) - 1.0
        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            x = self.act(x)
        x = self.fc[-1](x)
        return x
    
    
class PINN(nn.Module):
    def __init__(self, 
                 sub_layers, 
                 verbose,
                 lb,
                 ub,
                ):
        
        super(PINN, self).__init__()
        
        self.verbose = 0
        
        self.register_buffer('dummy', torch.tensor([]))
        
#         self.xlb = xlb
#         self.xub = xub
        

        # Initalize Neural Networks
        self.u_net = Net(sub_layers, lb, ub) 
        
    def load_pde_data(self, data):
        
#         # boundary points --- mse
#         # XPINN
#         #self.Xb1 = torch.tensor(Xb1, dtype=torch.float32)
#         self.xb1 = torch.unsqueeze(torch.tensor(xb1, dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
#         self.yb1 = torch.unsqueeze(torch.tensor(yb1, dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
#         self.ub1 = torch.unsqueeze(torch.tensor(ub1, dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
#         #self.Xb2 = torch.tensor(Xb2, dtype=torch.float32)
#         self.xb2 = torch.unsqueeze(torch.tensor(xb2, dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
#         self.yb2 = torch.unsqueeze(torch.tensor(yb2, dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
#         self.ub2 = torch.unsqueeze(torch.tensor(ub2, dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
        
#         # collocation points --- residual
#         # XPINN
#         self.xf1 = torch.unsqueeze(torch.tensor(xf1, dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
#         self.yf1 = torch.unsqueeze(torch.tensor(yf1, dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
#         self.ff1 = torch.unsqueeze(torch.tensor(ff1, dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
#         self.xf2 = torch.unsqueeze(torch.tensor(xf2, dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
#         self.yf2 = torch.unsqueeze(torch.tensor(yf2, dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
#         self.ff2 = torch.unsqueeze(torch.tensor(ff2, dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)

#         # interface points --- residual
#         self.xi1 = torch.unsqueeze(torch.tensor(xi1, dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
#         self.yi1 = torch.unsqueeze(torch.tensor(yi1, dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
#         self.fi1 = torch.unsqueeze(torch.tensor(fi1, dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
        
        
#         # testing pts
#         self.x_star1 = torch.unsqueeze(torch.tensor(x_star1, dtype=torch.float32),-1).to(self.dummy.device)
#         self.y_star1 = torch.unsqueeze(torch.tensor(y_star1, dtype=torch.float32),-1).to(self.dummy.device)
#         self.u_1 = torch.tensor(u_1, dtype=torch.float32).to(self.dummy.device)
        
#         self.x_star2 = torch.unsqueeze(torch.tensor(x_star2, dtype=torch.float32),-1).to(self.dummy.device)
#         self.y_star2 = torch.unsqueeze(torch.tensor(y_star2, dtype=torch.float32),-1).to(self.dummy.device)
#         self.u_2 = torch.tensor(u_2, dtype=torch.float32).to(self.dummy.device)


        # boundary points --- mse
        # XPINN
        #self.Xb1 = torch.tensor(Xb1, dtype=torch.float32)
        self.xb1 = torch.unsqueeze(torch.tensor(data['xb1'], dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
        self.yb1 = torch.unsqueeze(torch.tensor(data['yb1'], dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
        self.ub1 = torch.unsqueeze(torch.tensor(data['ub1'], dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
        #self.Xb2 = torch.tensor(Xb2, dtype=torch.float32)
        self.xb2 = torch.unsqueeze(torch.tensor(data['xb2'], dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
        self.yb2 = torch.unsqueeze(torch.tensor(data['yb2'], dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
        self.ub2 = torch.unsqueeze(torch.tensor(data['ub2'], dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
        
        # collocation points --- residual
        # XPINN
        self.xf1 = torch.unsqueeze(torch.tensor(data['xf1'], dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
        self.yf1 = torch.unsqueeze(torch.tensor(data['yf1'], dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
        self.ff1 = torch.unsqueeze(torch.tensor(data['ff1'], dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
        self.xf2 = torch.unsqueeze(torch.tensor(data['xf2'], dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
        self.yf2 = torch.unsqueeze(torch.tensor(data['yf2'], dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
        self.ff2 = torch.unsqueeze(torch.tensor(data['ff2'], dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)

        # interface points --- residual
        self.xi1 = torch.unsqueeze(torch.tensor(data['xi1'], dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
        self.yi1 = torch.unsqueeze(torch.tensor(data['yi1'], dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
        self.fi1 = torch.unsqueeze(torch.tensor(data['fi1'], dtype=torch.float32, requires_grad=True),-1).to(self.dummy.device)
        
        
        # testing pts
        self.x_star1 = torch.unsqueeze(torch.tensor(data['x1'], dtype=torch.float32),-1).to(self.dummy.device)
        self.y_star1 = torch.unsqueeze(torch.tensor(data['y1'], dtype=torch.float32),-1).to(self.dummy.device)
        self.u_1 = torch.tensor(data['u1'], dtype=torch.float32).to(self.dummy.device)
        
        self.x_star2 = torch.unsqueeze(torch.tensor(data['x2'], dtype=torch.float32),-1).to(self.dummy.device)
        self.y_star2 = torch.unsqueeze(torch.tensor(data['y2'], dtype=torch.float32),-1).to(self.dummy.device)
        self.u_2 = torch.tensor(data['u2'], dtype=torch.float32).to(self.dummy.device)
        
#         self.nu = data['nu']
#         self.mu = data['mu']
        self.beta = data['param']

        
#     def PDE(self, u, u_x, u_xx, u_y, u_yy, f): 
#         return u_xx + u_yy - f # Poisson problem residual
    
    def get_loss_pinn(self):
        
        xt_u_1 = torch.hstack([self.xb1, self.yb1])
        xt_u_2 = torch.hstack([self.xb2, self.yb2])
        xt_u = torch.vstack([xt_u_1, xt_u_2])
        u = torch.vstack([self.ub1, self.ub2])
        
        mse_u = (u-self.u_net(xt_u)).square().mean()
        
        x_f = torch.vstack([self.xf1, self.xf2])
        t_f = torch.vstack([self.yf1, self.yf2])
        xt_f = torch.hstack([x_f, t_f])
        
        #uf = self.u_net(xt_f).sum()
        uu = self.u_net(xt_f)
        uf = uu.sum()
        
        u_x = torch.autograd.grad(uf.sum(), x_f, create_graph=True)[0]
        u_y = torch.autograd.grad(uf.sum(), t_f, create_graph=True)[0]
        
        u_xx = torch.autograd.grad(u_x.sum(), x_f, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), t_f, create_graph=True)[0]
        
        u_xxx = torch.autograd.grad(u_xx.sum(), x_f, create_graph=True)[0]
        
        ff = torch.vstack([self.ff1, self.ff2])
        
        #mse_f = (u_xx + u_yy - ff).square().mean()
        #mse_f = (u_y-u_xx).square().mean() # Heat problem residual
        
        #print(uu.shape)
        #print(u_x.shape)
        #print(u_y.shape)
        #print(u_xx.shape)
        
        #mse_f = (u_y+self.mu*uu*u_x-self.nu*u_xx).square().mean()
        mse_f = (u_y+self.beta*u_x).square().mean()
        
        #loss = mse_u + mse_f
        
        loss = 20*mse_u + 1.*mse_f
        
        return loss
      
    def train_adam_pinn(self, epoch, adam_lr):
        
        params = self.u_net.parameters()
        
        optimizer = torch.optim.Adam(params, lr=adam_lr)
        
        for n in range(epoch):
            loss = self.get_loss_pinn()
            if n%100==0:
                if self.verbose == 1:
                    err = self.eval_pinn()
                    print('adam epoch %d, loss: %g, err=%f'%(n, loss.item(), err))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
    def train_lbfgs_pinn(self, lr, max_iter, max_eval, tolerance_grad, tolerance_change, history_size):
        params = self.u_net.parameters()
        
        optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=max_iter, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size)
        
        def closure():
            
            optimizer.zero_grad()
            loss = self.get_loss_pinn()
            loss.backward(retain_graph=True)
            
            global iter_count 
            iter_count += 1
            if iter_count%500 == 0:
                if self.verbose == 1:
                    err = self.eval_pinn()
                    print('lbfgs epoch %d, loss: %g, err=%f'%(iter_count, loss.item(), err))
            return loss
        optimizer.step(closure)
    
#     def predict_pinn(self, x_star1, y_star1, x_star2, y_star2):
#         #net1 = self.u1_xcgnet; net2 = self.u2_xcgnet
            
#         x_star1 = torch.unsqueeze(torch.tensor(x_star1, dtype=torch.float32),-1)
#         y_star1 = torch.unsqueeze(torch.tensor(y_star1, dtype=torch.float32),-1)
#         x_star2 = torch.unsqueeze(torch.tensor(x_star2, dtype=torch.float32),-1)
#         y_star2 = torch.unsqueeze(torch.tensor(y_star2, dtype=torch.float32),-1)
        
#         with torch.no_grad():
#             u1_pred = self.u_net(torch.cat((x_star1, y_star1), 1))
#             u2_pred = self.u_net(torch.cat((x_star2, y_star2), 1))
            
#         return u1_pred, u2_pred

    def predict_pinn(self, x_star1, y_star1, x_star2, y_star2):
        #net1 = self.u1_xcgnet; net2 = self.u2_xcgnet
            
        x_star1 = torch.unsqueeze(torch.tensor(x_star1, dtype=torch.float32),-1)
        y_star1 = torch.unsqueeze(torch.tensor(y_star1, dtype=torch.float32),-1)
        x_star2 = torch.unsqueeze(torch.tensor(x_star2, dtype=torch.float32),-1)
        y_star2 = torch.unsqueeze(torch.tensor(y_star2, dtype=torch.float32),-1)
        
        with torch.no_grad():
            u1_pred = self.u_net(torch.cat((x_star1, y_star1), 1))
            u2_pred = self.u_net(torch.cat((x_star2, y_star2), 1))
            
        return u1_pred, u2_pred
    
    def eval_pinn(self, ):
        
        #cprint('g', 'eval pinn kdv')
            
        with torch.no_grad():
            u1_pred = self.u_net(torch.cat((self.x_star1, self.y_star1), 1))
            u2_pred = self.u_net(torch.cat((self.x_star2, self.y_star2), 1))
            
            xcgpinn_u_pred1 = u1_pred.data.cpu().numpy()
            xcgpinn_u_pred2 = u2_pred.data.cpu().numpy()

            u_1 = self.u_1.data.cpu().numpy()
            u_2 = self.u_2.data.cpu().numpy()
            
            xcgpinn_u_pred = np.concatenate([xcgpinn_u_pred1, xcgpinn_u_pred2]).flatten()
            xcgpinn_u_vals = np.concatenate([u_1, u_2])
            
            xcgpinn_error = abs(xcgpinn_u_vals.flatten()-xcgpinn_u_pred.flatten())
            xcgpinn_error_u_total = np.linalg.norm(xcgpinn_u_vals.flatten()-xcgpinn_u_pred.flatten(),2)/np.linalg.norm(xcgpinn_u_vals.flatten(),2)
            
            return xcgpinn_error_u_total


class PINN_Client_Advec:
    
    def __init__(self, 
                 num_adam=None,
                 int_adam=None,
                 lr_adam=1e-3,
                 num_lbfgs=None,
                 int_lbfgs=None,
                 lr_lbfgs=1e-1,
                 layers=None,
                 device=torch.device('cuda:0'),
                 err_cap=100.0,
                ):
        
        self.device = device
        self.err_cap = err_cap
        
        self.num_adam = num_adam
        self.int_adam = int_adam
        self.lr_adam = lr_adam
        
        self.num_lbfgs = num_lbfgs
        self.int_lbfgs = int_lbfgs
        self.lr_lbfgs = lr_lbfgs
        

        self.layers = layers
        self.verbose = 0
        
        self.reset(param=10.0)
        
        self.breakpoint = 0
        self.cache = []
        
        
    def init_task_data(self, param):

        task_data = gen_advec_data(param)
        
        return task_data
  
    def train_adam(self,):
        
#         if interface is None:
#             raise Exception('Error, no interface given...')
            
        self.model.train_adam_pinn(
            self.int_adam, 
            self.lr_adam
        )
        
        err = self.model.eval_pinn()
        
        if np.isnan(err) or err > self.err_cap:
            warnings.warn('nan reward recerived, loss cap of {} applied...'.format(self.err_cap))
            err = self.err_cap
        
        return err
    
    def train_lbfgs(self,):
        
#         if interface is None:
#             raise Exception('Error, no interface given...')
            
        global iter_count 
        iter_count = 0
                       
        self.model.train_lbfgs_pinn(
            lr=self.lr_lbfgs, 
            max_iter=self.int_lbfgs, 
            max_eval=None, 
            tolerance_grad=1e-9, 
            tolerance_change=1e-12, 
            history_size=50
        )
        
        err = self.model.eval_pinn()
        
        if np.isnan(err) or err > self.err_cap:
            warnings.warn('nan reward recerived, loss cap of {} applied...'.format(self.err_cap))
            err = self.err_cap
        
        return err
    
    
    def query(self, ):
        
        #self.reset()
        
        hist_err = []
        
        for i_adam in range(self.num_adam):
            err = self.train_adam()
            #print(err)
            hist_err.append(err)
            
        for i_lbfgs in range(self.num_lbfgs):
            err = self.train_lbfgs()
            #print(err)
            hist_err.append(err)
        
        self.cache = hist_err.copy()
        hist_err = np.array(hist_err)
        
        return hist_err
    
    
    def step(self,):
        
        if self.breakpoint >= self.num_adam+self.num_lbfgs:
            warnings.warn('trunks limit {} is reached..., current num of breaks{}'.format(
                self.num_adam+self.num_lbfgs, self.breakpoint+1
            ))
            
        if self.breakpoint < self.num_adam:
            err = self.train_adam()
            #cprint('r', err)
        else:
            err = self.train_lbfgs()
            #cprint('b', err)
        #
        
        self.cache.append(err)
        self.breakpoint += 1
         
        return err

    def reset(self, param=None):
        
        self.pde_param = param
        
        if param is None:
            param = 10.0
            warnings.warn('no pde param used to initialize model, use {} as default'.format(param))
        
        cprint('y', 'switching to pde_param = {}'.format(param))
        self.task_data = self.init_task_data(param)
        
        xlb = min(self.task_data['xlb1'], self.task_data['xlb2'])
        xub = max(self.task_data['xub1'], self.task_data['xub2'])
        
        ylb = min(self.task_data['ylb1'], self.task_data['ylb2'])
        yub = max(self.task_data['yub1'], self.task_data['yub2'])
        
        lb = np.array([xlb, ylb])
        ub = np.array([xub, yub])

        self.model = PINN(sub_layers=self.layers, verbose=self.verbose, lb=lb, ub=ub).to(self.device)
        self.model.load_pde_data(self.task_data)
        
        self.cache = []
        self.breakpoint = 0
        
  
    
    
# pinn_client = PINN_Client_Advec(
#     num_adam=1,
#     int_adam=2000,
#     lr_adam=1e-3,
#     num_lbfgs=1,
#     int_lbfgs=50000,
#     lr_lbfgs=1e-1,
#     layers=[2,20,1],
#     device=torch.device('cuda:0'),
#     err_cap=100.0,
# )

# # pinn_client.train_adam()
# # pinn_client.train_lbfgs()
# pinn_client.reset(param=10)



# hist_err = pinn_client.query()
# print(hist_err)


# # pinn_client.reset(np.array([-1,-1,-1]))

# # hist_err = pinn_client.query()
# # print(hist_err)

