

import Fit_E_P_scikit_62_params as fit
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchsde import sdeint

line_lohman = np.load(r'Data/NGRIP_lohman_E_P.npy')

Ngrip_true = np.genfromtxt(r'Data/NGRIP_d18O_all_clean.txt', 
                        skip_header=70)


E_proccessed = torch.tensor(line_lohman[1][:-17][::20][::-1].copy())
P_proccessed = torch.tensor(line_lohman[3][:-17][::20][::-1].copy())
t_processed = torch.tensor(line_lohman[0][:-17][::20][::-1].copy())

E_proccessed /= E_proccessed.max()

A = 7.0e12 # m^2
secpryr = 3600.0 * 24.0 * 365.0
F0 = 3.0/secpryr # 3 m/yr/(sec/yr) = [m/s]
D = 1000.0 # m (depth of pycnocline)
tc = D/(2*F0*secpryr)
tc = D/(2*F0*secpryr)
N_times_1000 = 1
batch_size, state_size, t_size = 1_000, 2, len(E_proccessed) + 1000
ts = torch.linspace(0, 624, t_size) 
dt = ts[1]
sde = fit.SDE(t_size, dt)


sde.var = torch.nn.Parameter(torch.tensor([0.81200972,  1.33309881,  1.40519375,  2.23858595,  1.47614646, 1.96823767,  1.44208941,  2.10414714,  2.64022071,  1.41194388,2.494599  ,  1.18627672,  1.95762032,  1.40460393,  1.45426062,1.45213548,  1.19070766,  1.41634881,  1.2014333 ,  1.49512041,1.6488406 ,  1.47197924,  1.70183844,  1.41622403,  1.2367089 ,1.17885281,  1.38704138,  0.97722213,  0.4010249 ,  0.78949892,-0.53270506, -0.16716157, -0.26055302, -0.43184356, -0.21450713,-0.07040022, -0.42954994, -0.05424826, -0.60172172, -0.09041709,-0.13600629, -0.40348092, -0.47415006, -0.64554504, -0.26799165,-0.37117339, -0.05928456, -0.25065241,  0.03339209, -0.09458773,-0.42324827, -0.20758421, -0.55062141, -0.28542338, -0.15570241,-0.09979574, -0.39875957, -0.11978325, -1.24953184, -0.77326548]
),requires_grad = False)


sde.sigme = .2
y00 = torch.full(size=(1, batch_size), fill_value=1.0)
y01 = torch.full(size=(1, batch_size), fill_value=0.0)
y0 = torch.cat([y00, y01], dim=0)
sde.interpolate()

E_sim = torch.zeros((N_times_1000*batch_size, len(E_proccessed)))
P_sim = torch.zeros((N_times_1000*batch_size, len(E_proccessed)))
for i in range(N_times_1000):


    pred = sdeint(sde, y0, ts, method='srk',adaptive=False, dt=ts[1])
    is_stadial = fit.Crufix(pred[:,0,:])
    E_sim_iter = fit.E_calc(is_stadial)[:len(E_proccessed),:,0].T
    P_sim_iter = fit.P_calc(is_stadial)[:len(P_proccessed),:,0].T

    E_sim[i*batch_size:(i+1)*batch_size] = E_sim_iter
    P_sim[i*batch_size:(i+1)*batch_size] = P_sim_iter
    print(i+1, 'out of', N_times_1000, end='\r')

    E_sim = E_sim.detach().numpy().T
    P_sim = P_sim.detach().numpy().T

## plot the E_sim and P_sim, E_proccessed and P_proccessed

color_palette = sns.color_palette("colorblind")
fig, ax = plt.subplots(4,1, figsize=(20,20), sharex=True)
# set the x and y axis tick label size
fontsize = 20
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
# make the vspace between the subplots small
fig.subplots_adjust(hspace=0.05)
ts = np.linspace(0, 624, t_size)[::-1]
ts_plot = np.copy(t_processed)
ax[0].plot(ts_plot,E_sim[:,0]*12, label = 'E\'(t)', color = 'k', alpha = 0.5)
ax[0].plot(ts_plot,E_sim[:,1:]*12, color = 'k', alpha = 0.5)
ax[0].plot(ts_plot,E_proccessed.detach().numpy()*12, label = 'E(t)', color = color_palette[1])

ax[1].plot(ts_plot,P_sim[:,0],color = 'k', alpha = 0.5, label = 'P\'(t)')
ax[1].plot(ts_plot,P_sim[:,1:],color = 'k', alpha = 0.5)
ax[1].plot(ts_plot,P_proccessed.detach().numpy(), label = 'P(t)', color = color_palette[2])
ax[0].set_ylabel('E [events per 20 kyr]', fontsize=fontsize)
ax[1].set_ylabel('P', fontsize=fontsize)
ax[0].tick_params(axis='both', which='major', labelsize=fontsize)

# make time series with 20 points evenly spaced points over the 5200 points
ts_20 = np.linspace(0, 624, 20)*tc + 11000
ts_20 = ts_20[::-1]
ax[2].set_ylabel(r'$\gamma$', fontsize=fontsize)
ax[2].plot(ts*tc+11000,sde.gammas.detach().numpy()[0,0,:], label = r'$\gamma(t)$', color = color_palette[3])
# plot the individual gamma's
#ax[2].plot(ts_20,sde.gammas.detach().numpy()[0,0,::260], 'o', color = color_palette[3])
ax_22 = ax[2].twinx()
ax_22.plot(ts*tc+11000,sde.alphas.detach().numpy()[0,0,:], label = r'$\alpha(t)$', color = color_palette[4])
ax_22.set_ylabel(r'$\alpha$', fontsize=fontsize)
#flip the x axis0
ax[0].set_xlim(ax[0].get_xlim()[::-1])

print(ts[0]*tc+11000+10000, ts[-1]*tc+11000+10000)
ax[3].set_xlabel('Time [yr BP]', fontsize=fontsize)

# find the index with the best fit
RMSE_to_find = np.sqrt(np.mean((E_sim[:,0]- E_proccessed.detach().numpy())**2 + (P_sim[:,0]- P_proccessed.detach().numpy())**2))
min_rmse = np.argmin(RMSE_to_find)

ax[3].plot(ts*tc+11000, pred[:,0,min_rmse].detach().numpy(), label = 'Best fit', color = color_palette[0])
ax[3].fill_between(ts[1:]*tc+11000, 0, (1-is_stadial[1:,min_rmse].detach().numpy()-0.2)*3, color = 'grey', alpha = 0.5, label = 'Interstadial')
ax[3].set_ylim(0,2.2)

ax[0].legend(fontsize=fontsize)
ax[1].legend(fontsize=fontsize)
# get the labels from the ax_22
lines_1, labels_1 = ax[2].get_legend_handles_labels()
lines_2, labels_2 = ax_22.get_legend_handles_labels()
ax[2].legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=fontsize)
# ax[3].legend(fontsize=fontsize)

ax[3].set_xticks([121_000,111_000,101_000,91_000,81_000,71_000,61_000,51_000,41_000,31_000,21_000,11_000])
ax[3].set_xticklabels([121,111,101,91,81,71,61,51,41,31,21,11])
ax[3].set_ylabel(r'$\Delta$ b', fontsize=fontsize)

ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
ax[1].tick_params(axis='both', which='major', labelsize=fontsize)
ax[2].tick_params(axis='both', which='major', labelsize=fontsize)
ax[3].tick_params(axis='both', which='major', labelsize=fontsize)
ax_22.tick_params(axis='both', which='major', labelsize=fontsize)

x = 131_000
ax[0].text(x,14, '(a)', 
                fontsize=fontsize)
ax[1].text(x,0.8, '(b)',
                fontsize=fontsize)
ax[2].text(x,2.25, '(c)',
                fontsize=fontsize)
ax[3].text(x,2, '(d)',
                fontsize=fontsize)

ax[3].set_xlabel('Time [ka b2k]')
plt.savefig('Figures/Fit_alpha_20_gamma_20_sigma.png', bbox_inches='tight', dpi = 300)
plt.show()
