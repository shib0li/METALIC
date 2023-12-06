import numpy as np

import torch
import torch.nn as nn


class KernelRBF(nn.Module):
    
    def __init__(self, jitter=1e-5):
        super().__init__()     
        self.register_buffer('jitter', torch.tensor(jitter))
        self.log_amp = nn.Parameter(torch.tensor(0.0))
        self.log_ls = nn.Parameter(torch.tensor(0.0))
        
    def matrix(self, X):
        K = self.cross(X, X)
        Ijit = self.jitter*torch.eye(X.shape[0], device=self.jitter.device)
        K = K + Ijit
        return K

    def cross(self, X1, X2):
        norm1 = torch.reshape(torch.sum(torch.square(X1), dim=1), [-1,1])
        norm2 = torch.reshape(torch.sum(torch.square(X2), dim=1), [1,-1])        
        K = norm1-2.0*torch.matmul(X1,X2.T) + norm2
        K = torch.exp(-1.0*K/torch.exp(self.log_ls))
        return K
    
    
class KernelARD(nn.Module):
    def __init__(self, jitter=1e-5):
        super().__init__()     
        self.register_buffer('jitter', torch.tensor(jitter))
        self.log_amp = nn.Parameter(torch.tensor(0.0))
        self.log_ls = nn.Parameter(torch.tensor(0.0))
        
    def matrix(self, X):
        K = self.cross(X, X)
        Ijit = self.jitter*torch.eye(X.shape[0], device=self.jitter.device)
        K = K + Ijit
        return K

    def cross(self, X1, X2):
        norm1 = torch.reshape(torch.sum(torch.square(X1), dim=1), [-1,1])
        norm2 = torch.reshape(torch.sum(torch.square(X2), dim=1), [1,-1])        
        K = norm1-2.0*torch.matmul(X1,X2.T) + norm2
        K = torch.exp(self.log_amp)*torch.exp(-1.0*K/torch.exp(self.log_ls))
        return K
    
    
class KernelCat1(nn.Module):
    def __init__(self, jitter=1e-5):
        super().__init__()     
        self.register_buffer('jitter', torch.tensor(jitter))
        self.log_amp = nn.Parameter(torch.tensor(0.0))
        self.log_ls = nn.Parameter(torch.tensor(0.0))
        
    def matrix(self, X):
        K = self.cross(X, X)
        Ijit = self.jitter*torch.eye(X.shape[0], device=self.jitter.device)
        K = K + Ijit
        return K

    def cross(self, X1, X2):
        x = X1.unsqueeze(1)
        y = X2.unsqueeze(1).permute(1,0,2)
        dist = 1.0 - torch.abs(x - y)
        K = torch.exp(self.log_ls)*dist.mean(dim=2)
        K = torch.exp(K) 
        return K
    
class KernelCat2(nn.Module):
    def __init__(self, jitter=1e-5):
        super().__init__()     
        self.register_buffer('jitter', torch.tensor(jitter))
        self.log_amp = nn.Parameter(torch.tensor(0.0))
        self.log_ls = nn.Parameter(torch.tensor(0.0))
        
    def matrix(self, X):
        K = self.cross(X, X)
        Ijit = self.jitter*torch.eye(X.shape[0], device=self.jitter.device)
        K = K + Ijit
        return K

    def cross(self, X1, X2):
        x = X1.unsqueeze(1)
        y = X2.unsqueeze(1).permute(1,0,2)
        dist = torch.abs(x - y)
        K = (-torch.exp(self.log_ls))*dist.mean(dim=2)
        K = torch.exp(K) 
        return K
    
    
class KernelHybrid(nn.Module):
    def __init__(self, jitter=1e-5, dim_cont=1):
        super().__init__() 
        self.register_buffer('jitter', torch.tensor(jitter))
        self.dim_cont=dim_cont
        self.kernel_cont = KernelRBF(jitter)
        self.kernel_cate = KernelCat1(jitter)
        
    def matrix(self, X):
        K = self.cross(X, X)
        Ijit = self.jitter*torch.eye(X.shape[0], device=self.jitter.device)
        K = K + Ijit
        return K

    def cross(self, X1, X2):
        
        X1cont = X1[:, :self.dim_cont].reshape([-1,self.dim_cont])
        X2cont = X2[:, :self.dim_cont].reshape([-1,self.dim_cont])
        X1cat = X1[:, self.dim_cont:]
        X2cat = X2[:, self.dim_cont:]
        
        Kcont = self.kernel_cont.cross(X1cont, X2cont)
        Kcate = self.kernel_cate.cross(X1cat, X2cat)
        
        K = Kcont*Kcate
        
        return K
    
    

class KernelHybrid2(nn.Module):
    
    def __init__(self, jitter=1e-5, dim_cont1=1, dim_cont2=1):
        super().__init__() 
        
        self.register_buffer('jitter', torch.tensor(jitter))
        
        self.dim_cont1=dim_cont1
        self.dim_cont2=dim_cont2

        self.kernel_cont1 = KernelRBF(jitter)
        self.kernel_cont2 = KernelRBF(jitter)
        self.kernel_cate = KernelCat1(jitter)
        
    def matrix(self, X):
        K = self.cross(X, X)
        Ijit = self.jitter*torch.eye(X.shape[0], device=self.jitter.device)
        K = K + Ijit
        return K

    def cross(self, X1, X2):
        
        d1 = self.dim_cont1
        d2 = self.dim_cont2
        
        X1cont1 = X1[:, :d1].reshape([-1,d1])
        X2cont1 = X2[:, :d1].reshape([-1,d1])
        
        X1cont2 = X1[:, d1:d1+d2].reshape([-1,d2])
        X2cont2 = X2[:, d1:d1+d2].reshape([-1,d2])

        X1cat = X1[:, d1+d2:]
        X2cat = X2[:, d1+d2:]
        
        Kcont1 = self.kernel_cont1.cross(X1cont1, X2cont1)
        Kcont2 = self.kernel_cont2.cross(X1cont2, X2cont2)
        Kcate = self.kernel_cate.cross(X1cat, X2cat)
        
        K = Kcont1*Kcont2*Kcate
        
        return K
    
    
class KernelHybrid3(nn.Module):
    
    def __init__(self, jitter=1e-5, dim_cont1=1, dim_cont2=1, dim_cat1=9, dim_cat2=9):
        super().__init__() 
        
        self.register_buffer('jitter', torch.tensor(jitter))
        
        self.dim_cont1=dim_cont1
        self.dim_cont2=dim_cont2
        
        self.dim_cat1=dim_cat1
        self.dim_cat2=dim_cat2

        self.kernel_cont1 = KernelRBF(jitter)
        self.kernel_cont2 = KernelRBF(jitter)
        
        self.kernel_cat1 = KernelCat1(jitter)
        self.kernel_cat2 = KernelCat1(jitter)
        
    def matrix(self, X):
        K = self.cross(X, X)
        Ijit = self.jitter*torch.eye(X.shape[0], device=self.jitter.device)
        K = K + Ijit
        return K

    def cross(self, X1, X2):
        
#         print('kernel hybrid3333333')
        
        d1 = self.dim_cont1
        d2 = self.dim_cont2
        
        c1 = self.dim_cat1
        c2 = self.dim_cat2
        
        X1cont1 = X1[:, :d1].reshape([-1,d1])
        X2cont1 = X2[:, :d1].reshape([-1,d1])
        
        X1cont2 = X1[:, d1:d1+d2].reshape([-1,d2])
        X2cont2 = X2[:, d1:d1+d2].reshape([-1,d2])

        X1cat1 = X1[:, d1+d2:d1+d2+c1]
        X2cat1 = X2[:, d1+d2:d1+d2+c1]
        
        X1cat2 = X1[:, d1+d2+c1:]
        X2cat2 = X2[:, d1+d2+c1:]
        
        Kcont1 = self.kernel_cont1.cross(X1cont1, X2cont1)
        Kcont2 = self.kernel_cont2.cross(X1cont2, X2cont2)
        
        Kcat1 = self.kernel_cat1.cross(X1cat1, X2cat1)
        Kcat2 = self.kernel_cat2.cross(X1cat2, X2cat2)
        
        K = Kcont1*Kcont2*Kcat1*Kcat2
        
        return K
    
    
    