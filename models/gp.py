import numpy as np

import torch
import torch.nn as nn

from models.kernels import *

import warnings

from infras.misc import *

# def _psd_safe_cholesky(A, out=None, jitter=1e-5, max_tries=3):
    
#     if out is not None:
#         out = (out, torch.empty(A.shape[:-2], dtype=torch.int32, device=out.device))

#     L, info = torch.linalg.cholesky_ex(A, out=out)
#     if not torch.any(info):
#         return L

#     isnan = torch.isnan(A)
#     if isnan.any():
#         raise Exception('cholesky_cpu nan')
        
#     Aprime = A.clone()
#     jitter_prev = 0
#     for i in range(max_tries):
#         jitter_new = jitter * (10 ** i)
#         # add jitter only where needed
#         diag_add = ((info > 0) * (jitter_new - jitter_prev)).unsqueeze(-1).expand(*Aprime.shape[:-1])
#         Aprime.diagonal(dim1=-1, dim2=-2).add_(diag_add)
#         jitter_prev = jitter_new
#         warnings.warn(
#             f"A not p.d., added jitter of {jitter_new:.1e} to the diagonal"
#         )
#         L, info = torch.linalg.cholesky_ex(Aprime, out=out)
#         if not torch.any(info):
#             return L
        
#     raise Exception('Matrix not positive definite after repeatedly adding jitter')

# def psd_safe_cholesky(A, upper=False, out=None, jitter=1e-5, max_tries=3):
#     """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
#     Args:
#         :attr:`A` (Tensor):
#             The tensor to compute the Cholesky decomposition of
#         :attr:`upper` (bool, optional):
#             See torch.cholesky
#         :attr:`out` (Tensor, optional):
#             See torch.cholesky
#         :attr:`jitter` (float, optional):
#             The jitter to add to the diagonal of A in case A is only p.s.d. If omitted,
#             uses settings.cholesky_jitter.value()
#         :attr:`max_tries` (int, optional):
#             Number of attempts (with successively increasing jitter) to make before raising an error.
#     """
#     L = _psd_safe_cholesky(A, out=out, jitter=jitter, max_tries=max_tries)
#     if upper:
#         if out is not None:
#             out = out.transpose_(-1, -2)
#         else:
#             L = L.transpose(-1, -2)
#     return L

class GPR(nn.Module):
    
    def __init__(self, kernel='Hybrid', jitter=1e-5, dim_cont=1):
        super().__init__() 
        
        if kernel == 'RBF':
            self.kernel = KernelRBF(jitter)
        elif kernel == 'ARD':
            self.kernel = KernelARD(jitter)
        elif kernel == 'Cat1':
            self.kernel = KernelCat1(jitter)
        elif kernel == 'Cat2':
            self.kernel = KernelCat2(jitter)
        elif kernel == 'Hybrid':
            self.kernel = KernelHybrid(jitter, dim_cont)
        else:
            raise Exception('Error of kernel choice...')
            
            
        self.jitter = jitter
        
        self.log_tau = nn.Parameter(torch.tensor(0.0))
       
        self.register_buffer('dummy', torch.tensor([]))
        
        self.gamma_dist = torch.distributions.Gamma(100.0, 1.0)
        
    def forward(self, X, y, Xstar):
        
        Knn = self.kernel.matrix(X)
        Knm = self.kernel.cross(X, Xstar)
        Kmm = self.kernel.matrix(Xstar)
        
        Knn_noise = Knn + \
            torch.eye(X.shape[0]).to(self.dummy.device)/torch.exp(self.log_tau)
        
        L = torch.linalg.cholesky(Knn_noise)
        #L = psd_safe_cholesky(Knn_noise, jitter=self.jitter, max_tries=5)

        Linv_y = torch.linalg.solve(L, y)
        alpha = torch.linalg.solve(L.T, Linv_y)
        
        v = torch.linalg.solve(L, Knm)
        
        mu = torch.mm(Knm.T, alpha)
        V = Kmm - torch.mm(v.T, v)
        
        var = torch.diag(V).reshape(mu.shape)
        
        return mu, var
    
    
    def eval_nmllh(self, X, y, gamma_prior=True):
        
        #print(X)
        #cprint('y', y)
        #cprint('r', self.log_tau)
        
        Knn = self.kernel.matrix(X)
        
        Knn_noise = Knn + \
            torch.eye(X.shape[0]).to(self.dummy.device)/torch.exp(self.log_tau)
        
        L = torch.linalg.cholesky(Knn_noise)
        #L = psd_safe_cholesky(Knn_noise, jitter=self.jitter, max_tries=5)

        Linv_y = torch.linalg.solve(L, y)
        alpha = torch.linalg.solve(L.T, Linv_y)
        
        if gamma_prior:

            mllh = -0.5*torch.mm(y.T, alpha) -\
                   torch.sum(torch.log(torch.diag(L))) -\
                   0.5*X.shape[0]*np.log(2*np.pi) +\
                   self.gamma_dist.log_prob(torch.exp(self.log_tau))
        else:
            mllh = -0.5*torch.mm(y.T, alpha) -\
                   torch.sum(torch.log(torch.diag(L))) -\
                   0.5*X.shape[0]*np.log(2*np.pi)
        
        return -torch.sum(mllh)