print('Optimizing on sigma, alpha and gamma total upto 62 parameters')
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import  nn
import torchsde
from torchsde import sdeint

from scipy.interpolate import CubicSpline

import sys


NGRIP_data = np.genfromtxt('Data/NGRIP_d18O_all_clean.txt', delimiter='	', skip_header=70)
line_lohman = np.load(r'Data/NGRIP_lohman_E_P.npy')

E_max = torch.max(torch.tensor(line_lohman[1][::-1][:84000][::20].copy()))
E_proccessed = torch.tensor(line_lohman[1][::-1][:84000][::20].copy())/ E_max
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
    def interpolate(self):
        # Create time points for original data
        original_gamma_points = torch.linspace(0, 1, steps=self.gamma_res)
        original_alpha_points = torch.linspace(0, 1, steps=self.gamma_res)
        new_points = torch.linspace(0, 1, steps=self.t_size).cpu().numpy()

        # Create cubic spline for gamma
        gamma_spline = CubicSpline(original_gamma_points.cpu().numpy(), self.var[:self.gamma_res].cpu().detach().numpy())
        self.gammas = torch.tensor(gamma_spline(new_points), device=self.var.device).unsqueeze(0).unsqueeze(0)

        # Create cubic spline for alpha
        alpha_spline = CubicSpline(original_alpha_points.cpu().numpy(), self.var[self.gamma_res:].cpu().detach().numpy())
        self.alphas = torch.tensor(alpha_spline(new_points), device=self.var.device).unsqueeze(0).unsqueeze(0)
        

    def __init__(self, t_size, dt):
        super().__init__()
        self.var = nn.Parameter(torch.tensor( [
                1.2410061618840047, 1.2257541559477658, 1.74880024529583102, 2.594380465327546, 2.5, 
                1.707748764197484, 2.3950444980863441, 2.2109875536234263, 2.0453787737176428, 2.54923381014882837, 
                2.291261720048638, 2.29, 1.1448176849090805, 1.5410369878630763, 2.618312617855399, 
                1.3109067473965703, 1.195404400534151, 0.9735731458342838, 1.4834921133727313, 3.0063890141130274, 
                1.6963787504936398, 2.1436305178925616, 1.3068963407406016, 1.3030396744666248, 1.6787758033892834, 
                1.1222064423311608, 1.226184741779826, 0.9437848971213765, 0.993931413546175, 1.170634609371919, 
                -0.7471449007869402, -0.6492647497781524, -0.4692796654083996, -0.36138846286843773, -0.969401836176588, 
                -0.276358530842416, -0.5718077394272416, -0.28083000435222916, -0.5342302634025693, -0.19890563086473467, 
                -0.22199484395267688, -0.8975589062642788, -0.9445004855465486, -0.8465145072885629, -0.754905169356094, 
                -0.3756522110829975, -0.3576862083030943, -0.06158024007776863, -0.5600073222057935, -0.6736591981303011, 
                -0.10188241851784052, -0.911511129196867, -0.38770556726421446, -0.14466947812850828, -0.750715118322227, 
                -0.27570615571460555, -0.17116710519602107, -0.1315007592918631, -0.37830978548985605, -0.005251005965245414]
), requires_grad=False)
                

        #self.gamma = nn.Parameter(torch.tensor((1.35, 1.5, 1.35, 0.9, 1.2, 1.35, 1.2, 0.8, 1., 1.35)), requires_grad=True).expand(1, 1, 10) # Scalar parameter.
        
        self.t_size = t_size
        self.dt = dt
        self.gamma_res = 30
        self.alpha_res = 30
        self.sigma = 0.2
        
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        # update the interpolated array
        self.interpolate()




    def f(self, t, y):
        # index for the interpolated array
        index = int(t/self.dt)

        y0 = -y[1] - torch.abs(q0 + q1*(y[0] - b0))*(y[0] - b0)
        y1 = (y[0] + self.alphas[0,0,index] * y[1]  - self.gammas[0,0,index])/tau
        return torch.stack([y0, y1])
    
    def g(self, t, y):

        # make a mask for the sigma parameter that is batch_size x 1 to make
        # the different sigma values for different y[0] values possible
        mask = torch.where(y[0] < 1.1, torch.ones_like(y[0]), torch.zeros_like(y[0]))
        noice =  self.sigma * mask + self.sigma * (1 - mask)
        # expand noice to be batch_size x state_size
        expanded_noice = noice.unsqueeze(-1).expand(y.shape[1], y.shape[0]).T
        return expanded_noice



        


# consider vmap or jit but as of now it is not needed
# vmap: https://pytorch.org/tutorials/prototype/vmap_recipe.html
# jit:  https://pytorch.org/docs/stable/generated/torch.jit.script.html

"""
TODO:
    - Flip the is_stadial dimension 

"""
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
    rolling_mean /= E_max
    return rolling_mean


def rmse(is_stadial, E_data, P_data):
    E_sim = E_calc(is_stadial)[:4200,:,0]
    P_sim = P_calc(is_stadial)[:4200,:,0]
    # print the 4 mean values
    return torch.mean(torch.sqrt(torch.mean((E_sim - E_data)**2, dim = 0) + torch.mean((P_sim - P_data)**2, dim = 0)))


from skopt import gp_minimize

batch_size, state_size, t_size = 100, 2, 5_200
ts = torch.linspace(0, 624, t_size) 
dt = ts[1]
sde = SDE(t_size, dt)
y00 = torch.full(size=(1, batch_size), fill_value=1.3)
y01 = torch.full(size=(1, batch_size), fill_value=0.0)
y0 = torch.cat([y00, y01], dim=0)

def loss_func(var):
    sde.var = nn.Parameter(torch.tensor(var), requires_grad=False)  # Scalar parameter.
    sde.interpolate()
    pred = torchsde.sdeint(sde, y0, ts, method='srk',adaptive=False, dt=ts[1])
    is_stadial = Crufix(pred[:,0,:])
    
    loss = rmse(is_stadial, E_proccessed.unsqueeze(1), P_proccessed.unsqueeze(1))
    loos = np.round(loss,3)
    return loss.item()

init_guess =  [0.3, 0.3, 
        1.564408629735945, 1.5459925660984017, 1.8571771656284473, 1.640234950714301, 1.4232927358001546, 
        1.2063505208860081, 0.989408424123229, 0.7724663273604497, 1.0954624026158992, 1.63749233528977, 
        1.2553611363149142, 2.3338639192990915, 0.5805424863228205, 1.9215229525822266, 2.4084537256014387, 
        2.258594457425129, 1.9572636378310548, 1.3603701669883428, 2.254366647354426, 1.6185761455265342, 
        1.6434205000660334, 1.7756941606130419, 1.2657861683084106, 3.113211973289247, 2.4429345800971527, 
        1.6124338154938713, 1.1915996210739914, 2.0               , 2.0,               2.0, 
        -0.7795906773682628, -0.17346267876798227, -0.46742235221575823, -0.5143359542061089, -0.5612495561964594, 
        -0.6081631581868101, -0.6550767346270103, -0.7019903110672105, -0.4134029972698468, -0.5129494559062935, 
        -0.668664010762189, -0.27670138026728514, -0.23473368194649394, -0.555348440940895, -0.8329322779188049, 
        -0.6851068616625379, -0.4397472146928635, -0.25301278900774427, -0.7288284303889394, -0.8146190283227466, 
        -0.5414088385787968, -0.4995925947628611, -0.28313060284723135, -0.999               , -0.999, 
        -0.5292319633688891, -0.0001                , -0.2                , -0.2               ,-0.2]

# if main
if __name__ == '__main__':
    fit = True
    if fit:
        res = gp_minimize(loss_func, [(0.26, 0.32), (0.26, 0.32), # sigma
        (1.06, 2.06),   (1.05, 2.05),   (1.36, 2.36),   (1.14, 2.14),   (0.92, 1.92), 
        (0.71, 1.71),   (0.49, 1.49),   (0.27, 1.27),   (0.6, 1.6),     (1.14, 2.14),
        (0.76, 1.76),   (1.83, 2.83),   (0.08, 1.08),   (1.42, 2.42),   (1.91, 2.91), 
        (1.76, 2.76),   (1.46, 2.46),   (0.86, 1.86),   (1.75, 2.75),   (1.12, 2.12), 
        (1.14, 2.14),   (1.28, 2.28),   (0.77, 1.77),   (2.61, 3.61),   (1.94, 2.94), 
        (1.11, 2.11),   (0.69, 1.69),   (1.0, 2.0),     (2.5, 3.5),     (1.25, 2.25),
        (-0.93, -0.63), (-0.32, -0.02), (-0.62, -0.32), (-0.66, -0.36), (-0.71, -0.41), 
        (-0.76, -0.46), (-0.81, -0.51), (-0.85, -0.55), (-0.56, -0.26), (-0.66, -0.36),
        (-0.82, -0.52), (-0.43, -0.13), (-0.38, -0.08), (-0.71, -0.41), (-0.98, -0.68), 
        (-0.84, -0.54), (-0.59, -0.29), (-0.40, -0.10), (-0.88, -0.58), (-0.96, -0.66),
        (-0.69, -0.39), (-0.65, -0.35), (-0.43, -0.13), (-1, -0.7),     (-1, -0.7), 
        (-0.68, -0.38), (-0.3, 0.0), (-0.40, -0.10), (-0.30, 0.0), (-0.33, -0.02)
        ], n_calls=500, acq_optimizer = "lbfgs", n_jobs = 32, x0 = init_guess)
        print(res)
        print(res.x)
        print(res.fun)

    else:
        f_output = sde.f(10, y0)
        g_output = sde.g(10, y0)
        print('F and G output')
        print(f_output.shape)
        print(g_output.shape)
        # run the model with the starting parameters
        pred = torchsde.sdeint(sde, y0, ts, method='srk',adaptive=False, dt=ts[1])
        print('pred shape')
        print(pred.shape)
        is_stadial = Crufix(pred[:,0,:])
        E_sim = E_calc(is_stadial)[:4200,:,0]
        P_sim = P_calc(is_stadial)[:4200,:,0]

        # plot the results
        fig, ax = plt.subplots(2,1, figsize=(10,10))
        ax[0].plot(ts, pred[:,0,0].detach().numpy().T, label='E')

        ax[1].plot(ts[:4200], E_sim, label='E')
        ax_twin = ax[1].twinx()
        ax_twin.plot(ts[:4200], P_sim, label='P', color='orange')
        #plot the data
        ax[1].plot(ts[:4200], E_proccessed, label='E data', color='black', linestyle='--')
        ax_twin.plot(ts[:4200], P_proccessed, label='P data', color='black', linestyle='--')
        plt.show()

        # (0.26, 0.32), (0.26, 0.32), # sigma
        # (1.06, 2.06),   (1.05, 2.05),   (1.36, 2.36),   (1.14, 2.14),   (0.92, 1.92), 
        # (0.71, 1.71),   (0.49, 1.49),   (0.27, 1.27),   (0.6, 1.6),     (1.14, 2.14),
        # (0.76, 1.76),   (1.83, 2.83),   (0.08, 1.08),   (1.42, 2.42),   (1.91, 2.91), 
        # (1.76, 2.76),   (1.46, 2.46),   (0.86, 1.86),   (1.75, 2.75),   (1.12, 2.12), 
        # (1.14, 2.14),   (1.28, 2.28),   (0.77, 1.77),   (2.61, 3.61),   (1.94, 2.94), 
        # (1.11, 2.11),   (0.69, 1.69),   (1.0, 2.0),     (1.5, 3.5),     (1.25, 2.25),
        # (-0.93, -0.63), (-0.32, -0.02), (-0.62, -0.32), (-0.66, -0.36), (-0.71, -0.41), 
        # (-0.76, -0.46), (-0.81, -0.51), (-0.85, -0.55), (-0.56, -0.26), (-0.66, -0.36),
        # (-0.82, -0.52), (-0.43, -0.13), (-0.38, -0.08), (-0.71, -0.41), (-0.98, -0.68), 
        # (-0.84, -0.54), (-0.59, -0.29), (-0.40, -0.10), (-0.88, -0.58), (-0.96, -0.66),
        # (-0.69, -0.39), (-0.65, -0.35), (-0.43, -0.13), (-1, -0.7),     (-1, -0.7), 
        # (-0.68, -0.38), (-0.3, 0.0), (-0.40, -0.10), (-0.30, 0.0), (-0.33, -0.02)
