import matplotlib 
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns
import os

from multiprocessing import Pool

import torch
from torch import  nn
import torchsde
from torchsde import sdeint

E_data =  6.576929675080343
delta_E_data =  2.997816266913335
P_data =  0.44927676498507496
delta_P_data =  0.2297324573307872



color_palette = sns.color_palette("bright")

color_tot_points = 'k'
color_Ngrip = color_palette[6]
color_true_values = color_palette[4]



line_lohman = np.load(r'Data/NGRIP_lohman_E_P.npy')

E_proccessed = torch.tensor(line_lohman[1][::-1][:84000][::20].copy())
P_proccessed = torch.tensor(line_lohman[3][::-1][:84000][::20].copy())

## Constants to redimension the model
A = 7.0e12 # m^2
secpryr = 3600.0 * 24.0 * 365.0
F0 = 3.0/secpryr # 3 m/yr/(sec/yr) = [m/s]
D = 1000.0 # m (depth of pycnocline)
tc = D/(2*F0*secpryr)
b_c = 0.004 # charateristic buoyancy gradient
B_c = b_c*F0 # characteristic buoyancy flux

psi_0 = -4.5e6 # m^3/s
psi_1 = 20.0e6 # m^3/s¨
psi_a = 5.0e6 # m^3/s (5 Sv) 
chi = 2.5 # area scale factor



# Non-dimensional constants for the model
b0 = 0.625
q0 = -9.
q1 = 12.
tau = 0.902

class SDE(nn.Module):
    def __init__(self, t_size, dt):
        super().__init__()
        self.var = nn.Parameter(torch.tensor((0.2,1.5,0)), requires_grad=False)  # Scalar parameter.
        # list of numbers for gamma parameter to be interpolated

        #self.gamma = nn.Parameter(torch.tensor((1.35, 1.5, 1.35, 0.9, 1.2, 1.35, 1.2, 0.8, 1., 1.35)), requires_grad=True).expand(1, 1, 10) # Scalar parameter.
        
        self.t_size = t_size
        self.dt = dt

        
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        # update the interpolated array


    def f(self, t, y):
        # index for the interpolated array
        index = int(t/self.dt)

        y0 = -y[1] - torch.abs(q0 + q1*(y[0] - b0))*(y[0] - b0)
        y1 = (y[0] + self.var[2] * y[1]  - self.var[1])/tau
        return torch.stack([y0, y1])
    
    def g(self, t, y):
        return self.var[0] * torch.ones_like(y)


# consider vmap or jit but as of now it is not needed
# vmap: https://pytorch.org/tutorials/prototype/vmap_recipe.html
# jit:  https://pytorch.org/docs/stable/generated/torch.jit.script.html

def Crufix(y):
    time_events_out = torch.zeros_like(y)
    flag = torch.zeros_like(y[0])

    for i in range(y.shape[0]):
        flag = torch.where(y[i] < 1.0, torch.ones_like(flag), flag)
        flag = torch.where(y[i] > 1.4, torch.zeros_like(flag), flag)

        time_events_out[i] = flag    

    # copy the time_events_out and require grad
    return time_events_out

# finde the root mean difference between the simulation and the data

    
def P_calc(is_stadial, N = 1000):
    
    kernel = torch.ones((N,))/N
    rolling_mean = is_stadial.unfold(0, N, 1).matmul(kernel.view(-1,1))

    return rolling_mean

# define E_calc
def E_calc(is_stadial, N = 1000):
    diff = (( is_stadial - is_stadial.roll(1, dims=0)) > 0).float()
    
    kernel = torch.ones((N,))
    rolling_mean = diff.unfold(0, N, 1).matmul(kernel.view(-1,1))
    return rolling_mean

batch_size, state_size, t_size = 100, 2, 5_200
ts = torch.linspace(0, 624, t_size) 
dt = ts[1]
sde = SDE(t_size, dt)
y00 = torch.full(size=(1, batch_size), fill_value=1.0)
y01 = torch.full(size=(1, batch_size), fill_value=0.0)
y0 = torch.cat([y00, y01], dim=0)
sde.to('cpu')
sde.var.to('cpu')
def make_mean_E_P(var):
    sde.var = nn.Parameter(var, requires_grad=False)
    pred = torchsde.sdeint(sde, y0, ts, method='srk',adaptive=False, dt=ts[1])
    is_stadial = Crufix(pred[:,0])
    E = E_calc(is_stadial)
    P = P_calc(is_stadial)
    return E.mean(), P.mean(), E.std()/np.sqrt(batch_size), P.std()/np.sqrt(batch_size)

gammas = np.linspace(0.8,3.2, 75)
alphas = np.array([-0.0,-0.33, -0.66, -1.0])
sigmas = np.linspace(0.15, 0.4, 35)

# make a 3 grid of the parameters
sigma_grid, gamma_grid, alpha_grid = np.meshgrid(sigmas, gammas, alphas,indexing='ij')
tot_points = np.hstack([sigma_grid.flatten()[:,None], gamma_grid.flatten()[:,None], alpha_grid.flatten()[:,None]])
tot_points = torch.tensor(tot_points)


pool = Pool(8)
results = pool.map(make_mean_E_P, tot_points)

results = np.array(results)
np.save(f'results_sigmas_{sigmas.shape[0]}_gammas_{gammas.shape[0]}_alphas_{alphas.shape[0]}.npy', results)
np.save(f'tot_points_sigmas_{sigmas.shape[0]}_gammas_{gammas.shape[0]}_alphas_{alphas.shape[0]}.npy', tot_points)