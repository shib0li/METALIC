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


class pde_exact():
    def __init__(self):
        # 1D Burgers
        x, t, u, nu = sy.symbols('x, t, u, nu')
        m = 2*u*sy.sqrt(nu*t) # u substitution
        dmdu_conversion = 2*sy.sqrt(nu*t)
        integral_1 = sy.sin(sy.pi*(x-m))*sy.exp(-sy.cos(sy.pi*(x-m))/(2*sy.pi*nu))*dmdu_conversion
        integral_2 = sy.exp(-sy.cos(sy.pi*(x-m))/(2*sy.pi*nu))*dmdu_conversion
        # Hermite quadrature
        n = 100
        u_vals, w = np.polynomial.hermite.hermgauss(n)
        integral_1_sum = 0
        integral_2_sum = 0
        integral_total = 0
        for i in range(n):
            integral_1_sum += integral_1.subs(u,u_vals[i])*w[i]
            integral_2_sum += integral_2.subs(u,u_vals[i])*w[i]
        self.burgers_1_u = sy.lambdify([x, t, nu],integral_1_sum,'numpy')
        self.burgers_2_u = sy.lambdify([x, t, nu],integral_2_sum,'numpy')
        
    def burgers_sample(self, nu_val):
        x_val = np.linspace(-1,1,401)
        t_val = np.linspace(0,1,401)
        u_vals = np.zeros((x_val.size,t_val.size))
        for i in range(t_val.size): 
            if i == 0: # Take t = 0 as the initial condtions otherwise you divide by 0
                u_vals[:,i] = -np.sin(x_val*np.pi) # u(x, 0) = -sin(pi*x). 
            else:
                u_vals[:,i] = -(self.burgers_1_u(x_val,t_val[i],nu_val))/(self.burgers_2_u(x_val,t_val[i],nu_val))
        return u_vals.T, x_val, t_val

def gen_burgers_temp_spatio(
    nu=0.001,
    mu=1.0,
    x_lb=-1.0,
    x_ub=1.0,
    t_lb=0.0,
    t_ub=1.0,
    x_grid=501,
    t_grid=501,
):
    
    exact_data = pde_exact()
    
    u_vals, x_vals, t_vals = exact_data.burgers_sample(nu)
    
    #print(u_vals.shape)
    #print(x_vals.shape)
    #print(t_vals.shape)
    
    #print(x_vals)
    
    mesh_x, mesh_t = np.meshgrid(x_vals, t_vals)
    S = np.vstack((mesh_x.flatten(), mesh_t.flatten())).T
    V = u_vals.flatten().reshape([-1,1])
    u0 = u_vals[0,:]
    U = u_vals
    
    #print(S.shape)
    #print(V.shape)
    
    return S, V, u0, U
    
def gen_burgers_data(
    param,
    mu=1.0,
    Nu=100,
    Nf=1000,
):
    
    nu = param
    
    x_min=-1.0
    x_max=1.0
    
    t_min=0.0
    t_max=1.0
    
    margin = 0.1
    x_split_left = 0.0-margin
    x_split_right = 0.0+margin
    
    #print(x_split_left)
    #print(x_split_right)

    mesh, U, u0, _ = gen_burgers_temp_spatio(nu)
#     print(mesh.shape)
#     print(U.shape)
#     print(u0.shape)

    x_span = mesh[:,0]
    t_span = mesh[:,1]
    
#     print(x_span)
#     print(t_span)

    
    idx_bound_left = np.where(x_span==x_min)[0]
    idx_bound_right = np.where(x_span==x_max)[0]

    idx_bound_bottom = np.where(t_span==t_min)[0]
    idx_bound_top = np.where(t_span==t_max)[0]

#     idx_interface_left = np.where(x_span==x_split_left)[0]
#     idx_interface_right = np.where(x_span==x_split_right)[0]

    idx_interface_left = np.where(np.abs(x_span-x_split_left)<=0.001)[0]
    idx_interface_right = np.where(np.abs(x_span-x_split_right)<=0.001)[0]
    
    assert np.intersect1d(idx_interface_left, idx_interface_right).size == 0
    
    idx_interface = np.concatenate([idx_interface_left, idx_interface_right])
    
    #print(idx_interface_left.shape)
    #print(idx_interface_right.shape)

    #print(idx_bound_left.shape)
    #print(idx_bound_right.shape)

    #print(idx_bound_bottom.shape)
    #print(idx_bound_top.shape)


#     idx_bounds = idx_bound_left.tolist() +\
#                  idx_bound_right.tolist() +\
#                  idx_bound_bottom.tolist() +\
#                  idx_bound_top.tolist() +\
#                  idx_interface.tolist() 

    idx_bounds = idx_bound_left.tolist() +\
                 idx_bound_right.tolist() +\
                 idx_bound_bottom.tolist() +\
                 idx_bound_top.tolist() +\
                 idx_interface_left.tolist() +\
                 idx_interface_right.tolist()
    idx_bounds = np.array(list(set(idx_bounds)))

    idx_colls = np.delete(np.arange(mesh.shape[0]), idx_bounds)
    
    #print(idx_colls.size)
    #print(idx_bounds.size)
    #print(mesh.shape[0])
    
    assert idx_colls.size+idx_bounds.size == mesh.shape[0]
    
    # not include the top bound
    idx_bounds = idx_bound_left.tolist() +\
                 idx_bound_right.tolist() +\
                 idx_bound_bottom.tolist() +\
                 idx_interface_left.tolist() +\
                 idx_interface_right.tolist()
    idx_bounds = np.array(list(set(idx_bounds)))

    idx_region_1_part1 = np.where(x_span<x_split_left)[0]
    idx_region_1_part2 = np.where(x_span>x_split_right)[0]
    
    assert np.intersect1d(idx_region_1_part1, idx_region_1_part2).size == 0
    
    idx_region_1 = np.concatenate([idx_region_1_part1, idx_region_1_part2])
    #print(idx_region_1.shape)

    idx_colls1_part1 = np.intersect1d(idx_region_1_part1, idx_colls)
    idx_colls1_part2 = np.intersect1d(idx_region_1_part2, idx_colls)
    
    assert np.intersect1d(idx_colls1_part1, idx_colls1_part2).size == 0

    #print(idx_colls1_part1.shape)
    #print(idx_colls1_part2.shape)
    
    idx_colls1 = np.concatenate([idx_colls1_part1, idx_colls1_part2])
    

    idx_region_2_part1 = np.where(x_span>x_split_left)[0]
    idx_region_2_part2 = np.where(x_span<x_split_right)[0]
    
    idx_region_2 = np.intersect1d(idx_region_2_part1, idx_region_2_part2)
    
    #print(idx_region_2.shape)
    #print(idx_region_2)
    
    idx_colls2 = np.intersect1d(idx_region_2, idx_colls)
    #print(idx_colls2.shape)
    #print(idx_colls2)

    assert np.intersect1d(idx_colls1, idx_colls2).size == 0 # two collecations are disjoint
    
    #print(idx_colls1)
    #print(idx_colls2)
    #print(idx_colls1.shape)
    #print(idx_colls2.shape)

#     idx_bound_left1 = np.intersect1d(idx_region_1, idx_bound_left)
#     idx_bound_left2 = np.intersect1d(idx_region_2, idx_bound_left)
    
#     idx_bound_right1 = np.intersect1d(idx_region_1, idx_bound_right)
#     idx_bound_right2 = np.intersect1d(idx_region_2, idx_bound_right)

    idx_bound_left1 = np.intersect1d(idx_region_1, idx_bound_left)
    idx_bound_right1 = np.intersect1d(idx_region_1, idx_bound_right)
    idx_bound_bottom1 = np.intersect1d(idx_region_1, idx_bound_bottom)
    idx_bound_top1 = np.intersect1d(idx_region_1, idx_bound_top)
    
    #print(idx_bound_left1.shape)
    #print(idx_bound_right1.shape)
    #print(idx_bound_bottom1.shape)
    #print(idx_bound_top1.shape)
    
    idx_bound_bottom2 = np.intersect1d(idx_region_2, idx_bound_bottom)
    idx_bound_top2 = np.intersect1d(idx_region_2, idx_bound_top)
    
    #print(idx_bound_bottom2.shape)
    #print(idx_bound_top2.shape)
    
    idx_bound1 = np.hstack((
        idx_bound_left1,
        idx_bound_right1,
        idx_bound_bottom1,
        idx_bound_top1,
    ))
    
    #cprint('r', idx_bound_bottom.shape)
    #cprint('r', idx_bound_top.shape)
    
    idx_bound2 = np.hstack((
        idx_bound_bottom2,
        idx_bound_top2,
    ))
    
    #print(idx_bound1)
    #print(idx_bound2)
    #print(idx_bound1.shape)
    #print(idx_bound2.shape)


    sub_idx_bound1 = generate_random_choice(a=idx_bound1, N=Nu, seed=1)

    xb1 = x_span[sub_idx_bound1]
    yb1 = t_span[sub_idx_bound1]
    ub1 = U.squeeze()[sub_idx_bound1]
    
    sub_idx_bound2 = generate_random_choice(a=idx_bound2, N=Nu, seed=2, replace=True)

    xb2 = x_span[sub_idx_bound2]
    yb2 = t_span[sub_idx_bound2]
    ub2 = U.squeeze()[sub_idx_bound2]

    sub_idx_f1 = generate_random_choice(a=idx_colls1, N=Nf, seed=3)

    xf1 = x_span[sub_idx_f1]
    yf1 = t_span[sub_idx_f1]
    ff1 = U.squeeze()[sub_idx_f1]

    sub_idx_f2 = generate_random_choice(a=idx_colls2, N=Nf, seed=4)

    xf2 = x_span[sub_idx_f2]
    yf2 = t_span[sub_idx_f2]
    ff2 = U.squeeze()[sub_idx_f2]

    x1 = x_span[idx_region_1]
    y1 = t_span[idx_region_1]
    u1 = U.squeeze()[idx_region_1]

    x2 = x_span[idx_region_2]
    y2 = t_span[idx_region_2]
    u2 = U.squeeze()[idx_region_2]
    
    xi1 = x_span[idx_interface]
    yi1 = t_span[idx_interface]
    fi1 = U.squeeze()[idx_interface]


    #print(idx_bound_left.shape)
    #print(idx_bound_bottom1.shape)

    #print(idx_bound1.size)
    #print(idx_bound2.size)

    
    task_data = {}

    task_data['xb1'] = xb1
    task_data['yb1'] = yb1
    task_data['ub1'] = ub1

    task_data['xb2'] = xb2
    task_data['yb2'] = yb2
    task_data['ub2'] = ub2

    task_data['xf1'] = xf1
    task_data['yf1'] = yf1
    task_data['ff1'] = ff1

    task_data['xf2'] = xf2
    task_data['yf2'] = yf2
    task_data['ff2'] = ff2

    task_data['xi1'] = xi1
    task_data['yi1'] = yi1
    task_data['fi1'] = fi1

    task_data['x1'] = x1
    task_data['y1'] = y1
    task_data['u1'] = u1

    task_data['x2'] = x2
    task_data['y2'] = y2
    task_data['u2'] = u2
    
    task_data['nu'] = nu
    task_data['mu'] = mu
    
    task_data['x_span'] = x_span
    task_data['t_span'] = t_span
    task_data['u_span'] = U.squeeze()

    task_data['xlb1'] = x1.min()
    task_data['xub1'] = x1.max()
    task_data['xlb2'] = x2.min()
    task_data['xub2'] = x2.max()
    
    task_data['ylb1'] = y1.min()
    task_data['yub1'] = y1.max()
    task_data['ylb2'] = y2.min()
    task_data['yub2'] = y2.max()
    
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
        
        self.nu = data['nu']
        self.mu = data['mu']
        
#     def PDE(self, u, u_x, u_xx, u_y, u_yy, f): 
#         return u_xx + u_yy - f # Poisson problem residual

    def PDE(self, u, u_x, u_xx, u_y, u_yy, f): 
        
        #cprint('r', u.shape)
        #cprint('b', u_y.shape)
        #cprint('b', u_x.shape)
        #cprint('b', u_xx.shape)
        
        return u_y+self.mu*u*u_x-self.nu*u_xx # Burgers problem residual
    
    def get_loss_xcgpinn(self, i_weights):
        
        net1 = self.u1_xcgnet; net2 = self.u2_xcgnet
    
        mse_ub = 0; mse_f = 0; mse_i_u = 0; mse_i_uavg = 0; mse_i_res = 0; mse_i_res_continuity = 0; mse_i_flux = 0; 
        mse_i_uyy = 0; mse_i_gres = 0; mse_i_ux = 0; mse_i_uxx = 0;

        w_ub = 20; 
        w_f = 1; 
        w_interface = 5;

#         w_ub = 1; 
#         w_f = 1; 
#         w_interface = 1;
        
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
        
        #u1_xxx = torch.autograd.grad(u1_xx.sum(), self.xf1, create_graph=True)[0]
        
        # Sub-net 2  
        u2 = net2(torch.cat((self.xf2, self.yf2), 1))
        u2_sum = u2.sum()
        u2_x = torch.autograd.grad(u2_sum, self.xf2, create_graph=True)[0]
        u2_y = torch.autograd.grad(u2_sum, self.yf2, create_graph=True)[0]
        u2_xx = torch.autograd.grad(u2_x.sum(), self.xf2, create_graph=True)[0]
        u2_yy = torch.autograd.grad(u2_y.sum(), self.yf2, create_graph=True)[0]
        
        #u2_xxx = torch.autograd.grad(u2_xx.sum(), self.xf2, create_graph=True)[0]
        
        # Residuals 
        f1 = self.PDE(u1, u1_x, u1_xx, u1_y, u1_yy, self.ff1)
        f2 = self.PDE(u2, u2_x, u2_xx, u2_y, u2_yy, self.ff2)
        mse_f += f1.square().mean()
        mse_f += f2.square().mean()
        
        # Sub-net 1, Interface 1
        u1i1 = net1(torch.cat((self.xi1, self.yi1), 1))
        u1i1_sum = u1i1.sum()
        u1i1_x = torch.autograd.grad(u1i1_sum, self.xi1, create_graph=True)[0]
        u1i1_y = torch.autograd.grad(u1i1_sum, self.yi1, create_graph=True)[0]
        u1i1_xx = torch.autograd.grad(u1i1_x.sum(), self.xi1, create_graph=True)[0]
        u1i1_yy = torch.autograd.grad(u1i1_y.sum(), self.yi1, create_graph=True)[0]
        
        
        #u1i1_xxx = torch.autograd.grad(u1i1_xx.sum(), self.xi1, create_graph=True)[0]
        
        # Sub-net 2, Interface 1
        u2i1 = net2(torch.cat((self.xi1, self.yi1), 1))
        u2i1_sum = u2i1.sum()
        u2i1_x = torch.autograd.grad(u2i1_sum, self.xi1, create_graph=True)[0]
        u2i1_y = torch.autograd.grad(u2i1_sum, self.yi1, create_graph=True)[0]
        u2i1_xx = torch.autograd.grad(u2i1_x.sum(), self.xi1, create_graph=True)[0]
        u2i1_yy = torch.autograd.grad(u2i1_y.sum(), self.yi1, create_graph=True)[0]
        
        #u2i1_xxx = torch.autograd.grad(u2i1_xx.sum(), self.xi1, create_graph=True)[0]
        
        # Residual on the interfaces
        f1i1 = self.PDE(u1i1, u1i1_x, u1i1_xx, u1i1_y, u1i1_yy, self.fi1)
        f2i1 = self.PDE(u2i1, u2i1_x, u2i1_xx, u2i1_y, u2i1_yy, self.fi1) 
            
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
        #flux_i1 = u1i1_y - u2i1_y
        # (u1i1**2/2 - nu*u1i1_x) - (u2i1**2/2 - nu*u2i1_x)
        flux_i1 = (u1i1**2/2 - self.nu*u1i1_x) - (u2i1**2/2 - self.nu*u2i1_x)
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
        

class XPINN_Client_Burgers:
    
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
        
        self.reset(param=0.001)
        
        self.breakpoint = 0
        self.cache = []
        
        self.opts = self.num_adam  + self.num_lbfgs
        
        
    def init_task_data(self, param):
        
        task_data = gen_burgers_data(param)
        
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
    
    
    def query(self, interface=None):
        
        #self.reset()
        
        if interface.ndim == 1:
            interface = interface.reshape([1,-1])

        assert interface.shape[0] == 1 # only one interface could use in query mode
        
        hist_err = []
        
        for i_adam in range(self.num_adam):
            err = self.train_adam(interface.squeeze())
            #print(err)
            hist_err.append(err)
            
        for i_lbfgs in range(self.num_lbfgs):
            err = self.train_lbfgs(interface.squeeze())
            #print(err)
            hist_err.append(err)
        
        self.cache = hist_err.copy()
        hist_err = np.array(hist_err)
        
        return hist_err
    
    
    def batch(self, interface=None):
        
        #self.reset()
        
        if interface.ndim == 1:
            interface = interface.reshape([1,-1])
            
        assert interface.shape[0] == self.num_adam+self.num_lbfgs   
            
        hist_err = []
        
        for i_adam in range(self.num_adam):
            err = self.train_adam(interface[i_adam].squeeze())
            #print(err)
            #print(interface[i_adam].squeeze())
            hist_err.append(err)
            
        for i_lbfgs in range(self.num_lbfgs):
            err = self.train_lbfgs(interface[i_lbfgs+self.num_adam].squeeze())
            #print(err)
            #print(interface[i_lbfgs+self.num_adam].squeeze())
            hist_err.append(err)
        
        self.cache = hist_err.copy()
        hist_err = np.array(hist_err)
        
        return hist_err
        
    def step(self, interface=None):
        
        if self.breakpoint >= self.num_adam+self.num_lbfgs:
            warnings.warn('trunks limit {} is reached..., current num of breaks{}'.format(
                self.num_adam+self.num_lbfgs, self.breakpoint+1
            ))
        
        if interface.ndim == 1:
            interface = interface.reshape([1,-1])

        assert interface.shape[0] == 1 # only one interface could use in per step
            
        if self.breakpoint < self.num_adam:
            err = self.train_adam(interface.squeeze())
            #cprint('r', err)
        else:
            err = self.train_lbfgs(interface.squeeze())
            #cprint('b', err)
        #
        
        self.cache.append(err)
        self.breakpoint += 1
         
        return err

    def reset(self, param=None):
        
        self.pde_param = param
        
        if param is None:
            param = 0.001
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
        
  
    
    
# xpinn_client = XPINN_Client_Burgers(
#     num_adam=1,
#     int_adam=10000,
#     lr_adam=1e-3,
#     num_lbfgs=1,
#     int_lbfgs=100000,
#     lr_lbfgs=1e-1,
#     layers=[2,20,20,20,20,1],
#     device=torch.device('cuda:0'),
#     err_cap=100.0,
# )



# # i_weights = (np.random.uniform(size=[50, 9])>=0.5).astype(float)

# uni_weights = generate_with_bounds(
#     N=50,
#     lb=np.zeros(9),
#     ub=np.ones(9),
#     method='uniform',
#     seed=0
# )

# i_weights = (uni_weights>=0.5).astype(float)

# from tqdm.auto import tqdm, trange

# for i in tqdm(range(i_weights.shape[0])):
#     xpinn_client.reset(0.001)
#     wi = i_weights[i,:]
#     hist_err = xpinn_client.query(wi)
#     cprint('r', wi)
#     cprint('b', hist_err)
    
# # wi = i_weights[0,:]
# # hist_err = xpinn_client.query(wi)




