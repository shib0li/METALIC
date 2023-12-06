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

import warnings
from infras.randutils import *
from infras.misc import *

class Net(nn.Module):
    def __init__(self, layers, act=nn.Tanh()):
        super(Net, self).__init__()
        self.act = act
        self.fc = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.fc.append(nn.Linear(layers[i], layers[i+1]))
            nn.init.xavier_normal_(self.fc[-1].weight)
        
    def forward(self, x):
        for i in range(len(self.fc) - 1):
            x = self.fc[i](x)
            x = self.act(x)
        x = self.fc[-1](x)
        return x
    
    
class PINN(nn.Module):
    def __init__(self, 
                 sub_layers, 
                 verbose,
                ):
        
        super(PINN, self).__init__()
        
        self.verbose = verbose
        
        self.register_buffer('dummy', torch.tensor([]))
        

        # Initalize Neural Networks
        self.u_net = Net(sub_layers) 
        
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

        
    def PDE(self, u, u_x, u_xx, u_y, u_yy, f): 
        return u_xx + u_yy - f # Poisson problem residual
    
    def get_loss_pinn(self):
        
        xt_u_1 = torch.hstack([self.xb1, self.yb1])
        xt_u_2 = torch.hstack([self.xb2, self.yb2])
        xt_u = torch.vstack([xt_u_1, xt_u_2])
        u = torch.vstack([self.ub1, self.ub2])
        
        mse_u = (u-self.u_net(xt_u)).square().mean()
        
        x_f = torch.vstack([self.xf1, self.xf2])
        t_f = torch.vstack([self.yf1, self.yf2])
        xt_f = torch.hstack([x_f, t_f])
        
        uf = self.u_net(xt_f).sum()
        
        u_x = torch.autograd.grad(uf.sum(), x_f, create_graph=True)[0]
        u_y = torch.autograd.grad(uf.sum(), t_f, create_graph=True)[0]
        
        u_xx = torch.autograd.grad(u_x.sum(), x_f, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y.sum(), t_f, create_graph=True)[0]
        
        ff = torch.vstack([self.ff1, self.ff2])
        
        mse_f = (u_xx + u_yy - ff).square().mean()
        
#         loss = mse_u + mse_f

        loss = 20*mse_u + 1.*mse_f
        
        return loss
      
    def train_adam_pinn(self, epoch, adam_lr):
        
        params = self.u_net.parameters()
        
        optimizer = torch.optim.Adam(params, lr=adam_lr)
        
        for n in range(epoch):
            loss = self.get_loss_pinn()
            if n%100==0:
                if self.verbose == 1:
                    print('epoch %d, loss: %g'%(n, loss.item()))
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
                    print('epoch %d, loss: %g'%(iter_count, loss.item()))
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
        
        cprint('g', 'eval pinn poisson')
            
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
        

def square_erf(x,y, sharp):
    f = (erf(sharp*(x-.25))-erf(sharp*(x-0.75)))*(erf(sharp*(y-.25))-erf(sharp*(y-0.75)))
    return f

# http://www.vallis.org/salon2/lecture2-script.html
def forcing(grid, x, y, sharpness): 
    # Boundary values
    for i in range(x.shape[0]):
        for j in range(y.shape[0]):
            grid[i,j] = square_erf(x[i],y[j], sharpness)
    grid_normalize = grid.max()
    grid = grid/grid_normalize
    return grid, grid_normalize

def poisson_direct(gridsize, sharpness):
    A = np.zeros(shape=(gridsize,gridsize,gridsize,gridsize),dtype='d')
    b = np.zeros(shape=(gridsize,gridsize),dtype='d')
    x = np.linspace(0,1,gridsize)
    y = np.linspace(0,1,gridsize)
    dx = 1.0 / (gridsize - 1)
    
    # discretized differential operator
    for i in range(1,gridsize-1):
        for j in range(1,gridsize-1):
            A[i,j,i-1,j] = A[i,j,i+1,j] = A[i,j,i,j-1] = A[i,j,i,j+1] = 1/dx**2
            A[i,j,i,j] = -4/dx**2
    
    # boundary conditions
    for i in range(0, gridsize):
        A[0,i,0,i] = A[-1,i,-1,i] = A[i,0,i,0] = A[i,-1,i,-1] = 1
    
    # set the boundary values on the right side
    b, f_normalize = forcing(b, x, y, sharpness)

    return np.linalg.tensorsolve(A,b), b, x, y, f_normalize

def poisson_lhs(points, xmin, xmax, ymin, ymax, f_max_lhs, sharpness):
    loc = lhs(2, samples = points) # [0,1] x [0,1]
    x = xmin + (xmax-xmin)*loc[:,0]; y = ymin + (ymax-ymin)*loc[:,1]
    f = np.zeros((points))
    for i in range(points):
        f[i] = square_erf(x[i],y[i], sharpness)
    f = f/f_max_lhs
    return f, x, y


class PINN_Client_Poisson:
    
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
        
        self.reset(sharpness=1.0)
        
        self.breakpoint = 0
        self.cache = []

    def init_task_data(self, param):
        
        sharpness = param
        
        gridsize = 101

        u, f, x, y, f_max_lhs = poisson_direct(gridsize, sharpness)
        Y, X = np.meshgrid(x,y)

        y_interface = 0.5
        u1 = []; f1 = []; x1 = []; y1 = []; u2 = []; f2 = []; x2 = []; y2 = []; ui1 = []; fi1 = []; xi1 = []; yi1 = [] # Collocation points
        ub = []; fb = []; xb = []; yb = []; ub1 = []; fb1 = []; xb1 = []; yb1 = []; ub2 = []; fb2 = []; xb2 = []; yb2 = []; # Boundary points
        for i in range(gridsize):
            for j in range(gridsize):
                if y[j] > y_interface: # Top
                    u1.append(u[i,j]); f1.append(f[i,j]); x1.append(X[i,j]); y1.append(Y[i,j]);
                    if (x[i] == 0 and y[j] >= y_interface and y[j] <= 1) or (x[i] >= 0 and x[i] <= 1 and y[j] == 1) or (x[i] == 1 and y[j] >= y_interface and y[j] <= 1): # Outer Boundary
                        ub1.append(u[i,j]); fb1.append(f[i,j]); xb1.append(X[i,j]); yb1.append(Y[i,j]);
                if y[j] < y_interface: # Bottom
                    u2.append(u[i,j]); f2.append(f[i,j]); x2.append(X[i,j]); y2.append(Y[i,j]);
                    if (x[i] == 0 and y[j] >= 0 and y[j] <= y_interface) or (x[i] >= 0 and x[i] <= 1 and y[j] == 0) or (x[i] == 1 and y[j] >= 0 and y[j] <= y_interface): # Outer Boundary
                        ub2.append(u[i,j]); fb2.append(f[i,j]); xb2.append(X[i,j]); yb2.append(Y[i,j]);
                if y[j] == y_interface: # Interface
                        ui1.append(u[i,j]); fi1.append(f[i,j]); xi1.append(X[i,j]); yi1.append(Y[i,j]);

                if (x[i] == 0 and y[j] >= 0 and y[j] <= 1) or (x[i] == 1 and y[j] >= 0 and y[j] <= 1) or (x[i] >= 0 and x[i] <= 1 and y[j] == 0) or (x[i] >= 0 and x[i] <= 1 and y[j] == 1): # Outer Boundary
                    ub.append(u[i,j]); fb.append(f[i,j]); xb.append(X[i,j]); yb.append(Y[i,j]);


        u = u.flatten(); f = f.flatten(); x = X.flatten(); y = Y.flatten();
        u1 = np.array(u1); f1 = np.array(f1); x1 = np.array(x1); y1 = np.array(y1); u2 = np.array(u2); f2 = np.array(f2); x2 = np.array(x2); y2 = np.array(y2); 
        ui1 = np.array(ui1); fi1 = np.array(fi1); xi1 = np.array(xi1); yi1 = np.array(yi1); ub = np.array(ub); fb = np.array(fb); xb = np.array(xb); yb = np.array(yb); 
        ub1 = np.array(ub1); fb1 = np.array(fb1); xb1 = np.array(xb1); yb1 = np.array(yb1); ub2 = np.array(ub2); fb2 = np.array(fb2); xb2 = np.array(xb2); yb2 = np.array(yb2); 

        N_u = 100 # Number of boundary points
        N_f = 1000 # Number of collocation points

        # Collocation XPINN 1 selection
        ff1, xf1, yf1 = poisson_lhs(N_f, 0, 1, y_interface, 1, f_max_lhs, sharpness)

        # Collocation XPINN 2 selection
        ff2, xf2, yf2 = poisson_lhs(N_f, 0, 1, 0, y_interface, f_max_lhs, sharpness)

        # Boundary XPINN 1 selection
        idx = np.random.choice(ub1.shape[0], N_u, replace=False)
        ub1 = ub1[idx]; fb1 = fb1[idx]; xb1 = xb1[idx]; yb1 = yb1[idx];

        # Boundary XPINN 2 selection
        idx = np.random.choice(ub2.shape[0], N_u, replace=False)
        ub2 = ub2[idx]; fb2 = fb2[idx]; xb2 = xb2[idx]; yb2 = yb2[idx];\
        
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
            tolerance_grad=1e-6, 
            tolerance_change=1e-9, 
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

    def reset(self, sharpness=None):
        
        self.pde_param = sharpness
        
        if sharpness is None:
            warnings.warn('no sharpness used to initialize model, use sharpness=1.0 as default')
            sharpness = 1.0
        
        cprint('y', 'switching to sharpness = {}'.format(sharpness))
        self.task_data = self.init_task_data(sharpness)

        self.model = PINN(sub_layers=self.layers, verbose=self.verbose).to(self.device)
        self.model.load_pde_data(self.task_data)
        
        self.cache = []
        self.breakpoint = 0
        

# pinn_client = PINN_Client_Poisson(
#     num_adam=3,
#     int_adam=50,
#     lr_adam=1e-3,
#     num_lbfgs=4,
#     int_lbfgs=100,
#     lr_lbfgs=1e-1,
#     layers=[2,20,20,1],
#     device=torch.device('cuda:0'),
#     err_cap=100.0,
# )


# # pinn_client.train_adam()
# # pinn_client.train_lbfgs()


# # hist_err = pinn_client.query()
# # print(hist_err)

# # xpinn_client.reset(2.9)

# # hist_err = xpinn_client.batch(i_weights)
# # print(hist_err)

# # xpinn_client.reset(3.6)

# for i in range(3+4):
#     err = pinn_client.step()

# # xpinn_client.reset(3.8)
