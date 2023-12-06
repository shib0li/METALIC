import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time
import sympy as sy
import copy
from pyDOE import lhs
from scipy.special import erf

from scipy.integrate import odeint

import warnings
from infras.randutils import *
from infras.misc import *

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import time
import sympy as sy
import copy
from pyDOE import lhs
from scipy.special import erf

from scipy.integrate import odeint

import warnings
from infras.randutils import *
from infras.misc import *

def eval_h(x):
    nom = (x-np.pi)**2
    den = 2*((0.25*np.pi)**2)
    return np.exp(-nom/den)


def react_solver(x, t, param):
    
    rau = param
    
    if x.ndim == 1:
        x = x.reshape([-1,1])
        
    if t.ndim == 1:
        t = t.reshape([-1,1])
    
    hx = eval_h(x)
    
    nom = hx*np.exp(rau*t)
    den = hx*np.exp(rau*t) + 1 - hx
    
    return nom/den

def gen_reaction_data(
    param,
    Nu=100,
    Nf=1000,
):
    

    x_lb = 0.0
    x_ub = 2*np.pi

    t_lb = 0.0
    t_ub = 1.0

    x_grid=101
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

    colls1 = generate_with_bounds(N=Nf, lb=lb1, ub=ub1, method='lhs', seed=3)
    colls2 = generate_with_bounds(N=Nf, lb=lb2, ub=ub2, method='lhs', seed=4)
    
    num_i = 101
    
    interface = np.hstack((
        np.linspace(x_lb, x_ub, num_i).reshape([-1,1]),
        t_split*np.ones([num_i,1])
    ))
    
    #print(interface.shape)
    #print(interface)
    
    mesh_x, mesh_t = np.meshgrid(x, t)
    
    S = np.vstack((mesh_x.flatten(), mesh_t.flatten())).T
    U = react_solver(mesh_x.flatten(), mesh_t.flatten(), param)
    #print(U.shape)
    
    
    S1 = S[S[:,1]<t_split]
    S2 = S[S[:,1]>t_split]
    
    
    #print(S1.shape)
    #print(S2.shape)
    
    task_data = {}

    task_data['xb1'] = bound1[:,0]
    task_data['yb1'] = bound1[:,1]
    task_data['ub1'] = react_solver(bound1[:,0], bound1[:,1], param).squeeze()

    task_data['xb2'] = bound2[:,0]
    task_data['yb2'] = bound2[:,1]
    task_data['ub2'] = react_solver(bound2[:,0], bound2[:,1], param).squeeze()

    task_data['xf1'] = colls1[:,0]
    task_data['yf1'] = colls1[:,1]
    task_data['ff1'] = react_solver(colls1[:,0], colls1[:,1], param).squeeze()

    task_data['xf2'] = colls2[:,0]
    task_data['yf2'] = colls2[:,1]
    task_data['ff2'] = react_solver(colls2[:,0], colls2[:,1], param).squeeze()

    task_data['xi1'] = interface[:,0]
    task_data['yi1'] = interface[:,1]
    task_data['fi1'] = react_solver(interface[:,0], interface[:,1], param).squeeze()

    task_data['x1'] = S1[:,0]
    task_data['y1'] = S1[:,1]
    task_data['u1'] = react_solver(S1[:,0], S1[:,1], param).squeeze()
    
    #cprint('g', task_data['x1'].shape)

    task_data['x2'] = S2[:,0]
    task_data['y2'] = S2[:,1]
    task_data['u2'] = react_solver(S2[:,0], S2[:,1], param).squeeze()
    
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
    
    
class XPINN(nn.Module):
    def __init__(self, sub_layers, verbose, lb1, ub1, lb2, ub2):
        
        super(XPINN, self).__init__()
        
        self.verbose = verbose
        
        self.register_buffer('dummy', torch.tensor([]))


        # Initalize Neural Networks
        self.u1_xcgnet = Net(sub_layers, lb1, ub1)
        self.u2_xcgnet = Net(sub_layers, lb2, ub2)

        self.net_params_xcgpinn = list(self.u1_xcgnet.parameters()) + list(self.u2_xcgnet.parameters())
        
    def load_pde_data(self, data):
        
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
        self.rau = data['param']
        
#     def PDE(self, u, u_x, u_xx, u_y, u_yy, f): 
#         return u_xx + u_yy - f # Poisson problem residual

    def PDE(self, u, u_x, u_xx, u_y, u_yy, f, u_xxx): 
        
        #cprint('r', u.shape)
        #cprint('b', u_y.shape)
        #cprint('b', u_x.shape)
        #cprint('b', u_xx.shape)
        
        #return u_y+self.mu*u*u_x-self.nu*u_xx # Burgers problem residual
        
        #return u_y+self.beta*u_x
        
        return u_y - self.rau*u*(1-u)
    
    def get_loss_xcgpinn(self, i_weights):
        
        net1 = self.u1_xcgnet; net2 = self.u2_xcgnet
    
        mse_ub = 0; mse_f = 0; mse_i_u = 0; mse_i_uavg = 0; mse_i_res = 0; mse_i_res_continuity = 0; mse_i_flux = 0; 
        mse_i_uyy = 0; mse_i_gres = 0; mse_i_ux = 0; mse_i_uxx = 0;

        w_ub = 20; 
        w_f = 1; 
        w_interface = 5;
        
        if i_weights.ndim == 2:
            i_weights = i_weights.squeeze()

        w_i_u = i_weights[0]; 
        w_i_uavg = i_weights[1]; 
        w_i_res = i_weights[2]; 
        w_i_res_continuity = i_weights[3]; 
        w_i_flux = i_weights[4]; 
        w_i_uyy = i_weights[5]; 
        w_i_gres = i_weights[6]; 
        w_i_ux = i_weights[7]; 
        w_i_uxx = i_weights[8];

        
        # Boundry loss
        mse_ub += (self.ub1 - net1(torch.cat((self.xb1, self.yb1), 1))).square().mean() 
        mse_ub += (self.ub2 - net2(torch.cat((self.xb2, self.yb2), 1))).square().mean() 
        
        # Residual/collocation loss
        # Sub-net 1
        u1 = net1(torch.cat((self.xf1, self.yf1), 1))
        u1_sum = u1.sum()
        u1_x = torch.autograd.grad(u1_sum, self.xf1, create_graph=True)[0]
        u1_y = torch.autograd.grad(u1_sum, self.yf1, create_graph=True)[0]
        u1_xx = torch.autograd.grad(u1_x.sum(), self.xf1, create_graph=True)[0]
        u1_yy = torch.autograd.grad(u1_y.sum(), self.yf1, create_graph=True)[0]
        
        u1_xxx = torch.autograd.grad(u1_xx.sum(), self.xf1, create_graph=True)[0]
        
        # Sub-net 2  
        u2 = net2(torch.cat((self.xf2, self.yf2), 1))
        u2_sum = u2.sum()
        u2_x = torch.autograd.grad(u2_sum, self.xf2, create_graph=True)[0]
        u2_y = torch.autograd.grad(u2_sum, self.yf2, create_graph=True)[0]
        u2_xx = torch.autograd.grad(u2_x.sum(), self.xf2, create_graph=True)[0]
        u2_yy = torch.autograd.grad(u2_y.sum(), self.yf2, create_graph=True)[0]
        
        u2_xxx = torch.autograd.grad(u2_xx.sum(), self.xf2, create_graph=True)[0]
        
        # Residuals 
        f1 = self.PDE(u1, u1_x, u1_xx, u1_y, u1_yy, self.ff1, u1_xxx)
        f2 = self.PDE(u2, u2_x, u2_xx, u2_y, u2_yy, self.ff2, u2_xxx)
        mse_f += f1.square().mean()
        mse_f += f2.square().mean()
        
        # Sub-net 1, Interface 1
        u1i1 = net1(torch.cat((self.xi1, self.yi1), 1))
        u1i1_sum = u1i1.sum()
        u1i1_x = torch.autograd.grad(u1i1_sum, self.xi1, create_graph=True)[0]
        u1i1_y = torch.autograd.grad(u1i1_sum, self.yi1, create_graph=True)[0]
        u1i1_xx = torch.autograd.grad(u1i1_x.sum(), self.xi1, create_graph=True)[0]
        u1i1_yy = torch.autograd.grad(u1i1_y.sum(), self.yi1, create_graph=True)[0]
        
        
        u1i1_xxx = torch.autograd.grad(u1i1_xx.sum(), self.xi1, create_graph=True)[0]
        
        # Sub-net 2, Interface 1
        u2i1 = net2(torch.cat((self.xi1, self.yi1), 1))
        u2i1_sum = u2i1.sum()
        u2i1_x = torch.autograd.grad(u2i1_sum, self.xi1, create_graph=True)[0]
        u2i1_y = torch.autograd.grad(u2i1_sum, self.yi1, create_graph=True)[0]
        u2i1_xx = torch.autograd.grad(u2i1_x.sum(), self.xi1, create_graph=True)[0]
        u2i1_yy = torch.autograd.grad(u2i1_y.sum(), self.yi1, create_graph=True)[0]
        
        u2i1_xxx = torch.autograd.grad(u2i1_xx.sum(), self.xi1, create_graph=True)[0]
        
        # Residual on the interfaces
        f1i1 = self.PDE(u1i1, u1i1_x, u1i1_xx, u1i1_y, u1i1_yy, self.fi1, u1i1_xxx)
        f2i1 = self.PDE(u2i1, u2i1_x, u2i1_xx, u2i1_y, u2i1_yy, self.fi1, u2i1_xxx) 
            
        # Residual continuity conditions on the interfaces
        fi1 = f1i1 - f2i1
        mse_i_res += (f1i1).square().mean() 
        mse_i_res += (f2i1).square().mean() 
        mse_i_res_continuity += (fi1).square().mean() 
        
        # Average value (Required for enforcing the average solution along the interface)
        uavgi1 = (u1i1 + u2i1)/2  
        
        mse_i_u += (u1i1-u2i1).square().mean()
        mse_i_uavg += (u1i1-uavgi1).square().mean()
        mse_i_uavg += (u2i1-uavgi1).square().mean()
        
        # Flux continuity conditions on the interfaces
        flux_i1 = u1i1_y - u2i1_y
        mse_i_flux += (flux_i1).square().mean()
        
        # GPINN continuity residual
        f1i1_x = torch.autograd.grad(f1i1.sum(), self.xi1, create_graph=True)[0]
        f1i1_y = torch.autograd.grad(f1i1.sum(), self.yi1, create_graph=True)[0]
        f2i1_x = torch.autograd.grad(f2i1.sum(), self.xi1, create_graph=True)[0]
        f2i1_y = torch.autograd.grad(f2i1.sum(), self.yi1, create_graph=True)[0]
        
        mse_i_gres += (f1i1_x).square().mean() + (f1i1_y).square().mean()
        mse_i_gres += (f2i1_x).square().mean() + (f2i1_y).square().mean()
        
        # u_yy continuity
        mse_i_uyy += (u1i1_yy - u2i1_yy).square().mean()
        
        # u_x and u_xx continuity
        mse_i_ux += (u1i1_x - u2i1_x).square().mean()
        mse_i_uxx += (u1i1_xx - u2i1_xx).square().mean()

        net_loss = w_ub*mse_ub + w_f*mse_f
        interface_loss = w_i_u*mse_i_u + w_i_uavg*mse_i_uavg + w_i_res*mse_i_res + w_i_res_continuity*mse_i_res_continuity + w_i_flux*mse_i_flux + w_i_gres*mse_i_gres
        interface_loss += w_i_uyy*mse_i_uyy + w_i_ux*mse_i_ux + w_i_uxx*mse_i_uxx
        
        loss = net_loss + w_interface*interface_loss
        return loss
      
    def train_adam_xcgpinn(self, epoch, adam_lr, interface):
        
        params = self.net_params_xcgpinn 
        
        optimizer = torch.optim.Adam(params, lr=adam_lr)
        
        for n in range(epoch):
            loss = self.get_loss_xcgpinn(i_weights=interface)
            if n%100==0:
                if self.verbose == 1:
                    err = self.eval_xcgpinn()
                    print('adam epoch %d, loss: %g, err=%f'%(n, loss.item(), err))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
     
        return 
        
    def train_lbfgs_xcgpinn(self, lr, max_iter, max_eval, tolerance_grad, tolerance_change, history_size, interface):
        params = self.net_params_xcgpinn 
        optimizer = torch.optim.LBFGS(params, lr=lr, max_iter=max_iter, max_eval=max_eval,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change, history_size=history_size)
        def closure():
            
            optimizer.zero_grad()
            loss = self.get_loss_xcgpinn(i_weights=interface)
            loss.backward(retain_graph=True)
            
            global iter_count 
            iter_count += 1
            if iter_count%500 == 0:
                if self.verbose == 1:
                    err = self.eval_xcgpinn()
                    print('lbfgs epoch %d, loss: %g, err=%f'%(iter_count, loss.item(), err))
            return loss
        optimizer.step(closure)
    
    def predict_xcgpinn(self, x_star1, y_star1, x_star2, y_star2):
        net1 = self.u1_xcgnet; net2 = self.u2_xcgnet
            
        x_star1 = torch.unsqueeze(torch.tensor(x_star1, dtype=torch.float32),-1).to(self.dummy.device)
        y_star1 = torch.unsqueeze(torch.tensor(y_star1, dtype=torch.float32),-1).to(self.dummy.device)
        x_star2 = torch.unsqueeze(torch.tensor(x_star2, dtype=torch.float32),-1).to(self.dummy.device)
        y_star2 = torch.unsqueeze(torch.tensor(y_star2, dtype=torch.float32),-1).to(self.dummy.device)
        with torch.no_grad():
            u1_pred = net1(torch.cat((x_star1, y_star1), 1))
            u2_pred = net2(torch.cat((x_star2, y_star2), 1))
        return u1_pred, u2_pred
    
    def eval_xcgpinn(self, ):
        
        #cprint('g', 'eval xpinn kdv')
        
        net1 = self.u1_xcgnet; net2 = self.u2_xcgnet
            
        with torch.no_grad():
            u1_pred = net1(torch.cat((self.x_star1, self.y_star1), 1))
            u2_pred = net2(torch.cat((self.x_star2, self.y_star2), 1))
            
            xcgpinn_u_pred1 = u1_pred.data.cpu().numpy()
            xcgpinn_u_pred2 = u2_pred.data.cpu().numpy()
            
            
            
            u_1 = self.u_1.data.cpu().numpy()
            u_2 = self.u_2.data.cpu().numpy()
            
            xcgpinn_u_pred = np.concatenate([xcgpinn_u_pred1, xcgpinn_u_pred2]).flatten()
            xcgpinn_u_vals = np.concatenate([u_1, u_2])
            
            xcgpinn_error = abs(xcgpinn_u_vals.flatten()-xcgpinn_u_pred.flatten())
            xcgpinn_error_u_total = np.linalg.norm(xcgpinn_u_vals.flatten()-xcgpinn_u_pred.flatten(),2)/np.linalg.norm(xcgpinn_u_vals.flatten(),2)
            
            return xcgpinn_error_u_total
        

class XPINN_Client_Reaction:
    
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
        
        assert num_adam == num_lbfgs
        
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
        
        self.reset(param=5.0)
        
        self.breakpoint = 0
        self.cache = []
        
        self.opts = self.num_adam
        
        
    def init_task_data(self, param):
        
        task_data = gen_reaction_data(param)
        
        return task_data
    
  
    def train_adam(self, interface=None):
        
        if interface is None:
            raise Exception('Error, no interface given...')
            
        self.model.train_adam_xcgpinn(
            self.int_adam, 
            self.lr_adam, 
            interface
        )
        
        err = self.model.eval_xcgpinn()
        
        if np.isnan(err) or err > self.err_cap:
            warnings.warn('nan reward recerived, loss cap of {} applied...'.format(self.err_cap))
            err = self.err_cap
        
        return err
    
    def train_lbfgs(self, interface=None):
        
        if interface is None:
            raise Exception('Error, no interface given...')
            
        global iter_count 
        iter_count = 0
                       
        self.model.train_lbfgs_xcgpinn(
            lr=self.lr_lbfgs, 
            max_iter=self.int_lbfgs, 
            max_eval=None, 
            tolerance_grad=1e-9, 
            tolerance_change=1e-12, 
            history_size=50,
            interface=interface
        )
        
        err = self.model.eval_xcgpinn()
        
        if np.isnan(err) or err > self.err_cap:
            warnings.warn('nan reward recerived, loss cap of {} applied...'.format(self.err_cap))
            err = self.err_cap
        
        return err
    
    
    def unit_step(self, interface=None):
        
        if interface.ndim == 1:
            interface = interface.reshape([1,-1])

        assert interface.shape[0] == 1 # only one interface could use in unit mode
        
        err1 = self.train_adam(interface.squeeze())
        err2 = self.train_lbfgs(interface.squeeze())
        
        err = np.array([err1, err2])

        return err[-1]
        
        
#         hist_err = []
        
#         for i_adam in range(self.num_adam):
#             err = self.train_adam(interface.squeeze())
#             #print(err)
#             hist_err.append(err)
            
#         for i_lbfgs in range(self.num_lbfgs):
#             err = self.train_lbfgs(interface.squeeze())
#             #print(err)
#             hist_err.append(err)
        
#         self.cache = hist_err.copy()
#         hist_err = np.array(hist_err)
        
#         return hist_err
    
    
    def query(self, interface=None):
        
        #self.reset()
        
        #cprint('r', 'stage query')
        
        if interface.ndim == 1:
            interface = interface.reshape([1,-1])

        assert interface.shape[0] == 1 # only one interface could use in query mode
        
        hist_err = []
        
        for i in range(self.opts):
            err = self.unit_step(interface.squeeze())
            #print(err)
            hist_err.append(err)
        
        self.cache = hist_err.copy()
        hist_err = np.array(hist_err)
        
        return hist_err
    
    
    def batch(self, interface=None):
        
        #self.reset()
        
        #cprint('r', 'stage batch')
        
        if interface.ndim == 1:
            interface = interface.reshape([1,-1])
            
        assert interface.shape[0] == self.opts 
            
        hist_err = []
        
        for i in range(self.opts):
            err = self.unit_step(interface[i,:].squeeze())
            #cprint('r', err)
            #cprint('b', interface[i,:].squeeze())
            hist_err.append(err)
        
        self.cache = hist_err.copy()
        hist_err = np.array(hist_err)
        
        return hist_err
        
    def step(self, interface=None):
        
        #cprint('r', 'stage step')
        
        if self.breakpoint >= self.opts:
            warnings.warn('trunks limit {} is reached..., current num of breaks{}'.format(
                self.opts, self.breakpoint+1
            ))
        
        if interface.ndim == 1:
            interface = interface.reshape([1,-1])

        assert interface.shape[0] == 1 # only one interface could use in per step
            
#         if self.breakpoint < self.num_adam:
#             err = self.train_adam(interface.squeeze())
#             #cprint('r', err)
#         else:
#             err = self.train_lbfgs(interface.squeeze())
#             #cprint('b', err)
#         #

        err = self.unit_step(interface.squeeze())
        #cprint('b', err)
        
        self.cache.append(err)
        self.breakpoint += 1
         
        return err

    def reset(self, param=None):
        
        self.pde_param = param
        
        if param is None:
            param = 5.0
            warnings.warn('no pde param used to initialize model, use {} as default'.format(param))
        
        cprint('y', 'switching to pde_param = {}'.format(param))
        self.task_data = self.init_task_data(param)
        
        
        lb1 = np.array([self.task_data['xlb1'], self.task_data['ylb1']])
        ub1 = np.array([self.task_data['xub1'], self.task_data['yub1']])
        
        #print(lb1, ub1)
        
        lb2 = np.array([self.task_data['xlb2'], self.task_data['ylb2']])
        ub2 = np.array([self.task_data['xub2'], self.task_data['yub2']])
        
        #print(lb2, ub2)
    
    

        self.model = XPINN(
            sub_layers=self.layers, 
            verbose=self.verbose, 
            lb1=lb1, 
            ub1=ub1, 
            lb2=lb2, 
            ub2=ub2,
        ).to(self.device)
        self.model.load_pde_data(self.task_data)
        
        self.cache = []
        self.breakpoint = 0
        

    
    
# xpinn_client = XPINN_Client_Reaction(
#     num_adam=2,
#     int_adam=1000,
#     lr_adam=1e-3,
#     num_lbfgs=2,
#     int_lbfgs=2000,
#     lr_lbfgs=1e-1,
#     layers=[2,20,1],
#     device=torch.device('cuda:0'),
#     err_cap=100.0,
# )

# i_weights = (np.random.uniform(size=[3+4,9])>=0.5).astype(float)

# print(i_weights)


# # xpinn_client.train_adam(interface=i_weights[0,:])
# # xpinn_client.train_lbfgs(interface=i_weights[1,:])


# hist_err = xpinn_client.query(i_weights[0,:])
# print(hist_err)

# # xpinn_client.reset(2.9)

# # hist_err = xpinn_client.batch(i_weights)
# # print(hist_err)

# # xpinn_client.reset(3.6)

# # for i in range(3+4):
# #     err = xpinn_client.step(i_weights[i,:])

# # xpinn_client.reset(3.8)



