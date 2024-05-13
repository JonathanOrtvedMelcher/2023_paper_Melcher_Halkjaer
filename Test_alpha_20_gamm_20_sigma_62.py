

import Fit_E_P_scikit_62_params_last as fit
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torchsde import sdeint

line_lohman = np.load(r'Data/NGRIP_lohman_E_P.npy')

Ngrip_true = np.genfromtxt(r'Data/NGRIP_d18O_all_clean.txt', 
                        skip_header=70)


E_proccessed = torch.tensor(line_lohman[1][::-1][:84000][::20].copy())
P_proccessed = torch.tensor(line_lohman[3][::-1][:84000][::20].copy())
E_proccessed /= E_proccessed.max()

A = 7.0e12 # m^2
secpryr = 3600.0 * 24.0 * 365.0
F0 = 3.0/secpryr # 3 m/yr/(sec/yr) = [m/s]
D = 1000.0 # m (depth of pycnocline)
tc = D/(2*F0*secpryr)
tc = D/(2*F0*secpryr)
N_times_1000 = 1
batch_size, state_size, t_size = 1_000, 2, 5_200
ts = torch.linspace(0, 624, t_size) 
dt = ts[1]
sde = fit.SDE(t_size, dt)


sde.var = nn.Parameter(torch.tensor( [
    0.9656540431571357, 1.4423091428527133, 1.264132277175128, 2.290897522347719, 1.9298199813437038, 
    1.5636742064439269, 2.196675217040514, 1.4001705164006417, 2.4305639399969334, 2.2447910051845317, 
    1.516883265854013, 2.0739572218681235, 1.04840963, 1.70094475, 1.5792577, 
    1.47990231, 1.4882681182767747, 1.1829733212599782, 1.356321891845398, 1.3226841275743053, 
    1.4116099843974599, 1.6442123223740315, 1.3532412112229366,   1.756032554643457, 1.3674355 , 
    1.29002962,  1.22850877,  1.57792757,  1.02694072,  1.14734267, 
    -0.49219827, -0.09538826, -0.17897563, -0.49370938, -0.10874845, 
    -0.42945039, -0.12312627, -0.15570303, -0.24837356, -0.7009623 , 
    -0.09987886, -0.0576511 , -0.26419443, -0.35388991, -0.7671657 , 
    -0.32114277, -0.37117339, -0.04164061, -0.23075913,  0.01132729, 
    -0.0329518 , -0.40050239, -0.22401868, -0.61337606, -0.20483656, 
    -0.22174097, -0.14455042, -0.56811131, -0.36302431, -0.75605609]
), requires_grad=False)

# sde.var = torch.nn.Parameter(torch.tensor(
#         [0.781079923758534, 1.4978214976590256, 1.6730786682672487, 2.428036755250182, 2.0088359898143486, 1.6929960671892939, 2.5657980852909743, 1.6280619268941467, 2.2628351021689657, 2.4908354226763296, 1.588099881819318, 1.7656851559400304, 0.7457957791998361, 2.1216472038068384, 2.543686477564782, 0.8230037157158218, 1.5893833177855423, 1.3133664772615632, 1.5168451838472303, 1.5887404100763833, 1.69862909540113, 2.2711067417240347, 1.4474192613924362, 1.1442594676328186, 1.8816924171597562, 1.2484894403830995, 1.4693974637357052, 1.759343290966731, 0.7702166340367896, 1.4546897738420006, -0.7226827424023656, -0.6441412344475707, -0.14231352440956302, -0.49386802405098573, -0.06766360674877014, -0.592587971241957, -0.6306183295979957, -0.09594532902198938, -0.5086846558987949, -0.4922130259204907, -0.2916649347252684, -0.06054009424523332, -0.7774191010227092, -0.8864426735908968, -0.999, -0.7256522110829975, -0.5259225089358215, -0.18529210830842638, -0.25643446337914544, -0.10996003296184662, -0.11154722351065205, -0.999, -0.18396496258373518, -0.38761625741131617, -0.6690903119896459, -0.0370357979707596, -0.5913273034008911, -0.48273598314791943, -0.6592044772664949, -0.31506160625073154]
# ),requires_grad = False)
sde.sigme = .2
y00 = torch.full(size=(1, batch_size), fill_value=1.0)
y01 = torch.full(size=(1, batch_size), fill_value=0.0)
y0 = torch.cat([y00, y01], dim=0)
sde.interpolate()

E_sim = torch.zeros((N_times_1000*batch_size, 4200))
P_sim = torch.zeros((N_times_1000*batch_size, 4200))
for i in range(N_times_1000):


    pred = sdeint(sde, y0, ts, method='srk',adaptive=False, dt=ts[1])
    is_stadial = fit.Crufix(pred[:,0,:])
    E_sim_iter = fit.E_calc(is_stadial)[:4200,:,0].T
    P_sim_iter = fit.P_calc(is_stadial)[:4200,:,0].T

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
ts_plot = ts[:4200]*tc
ax[0].plot(ts_plot,E_sim[:,0]*12, label = 'E\'(t)', color = 'k', alpha = 0.5)
ax[0].plot(ts_plot,E_sim[:,1:]*12, color = 'k', alpha = 0.5)
ax[0].plot(ts_plot,E_proccessed.detach().numpy()*12, label = 'E(t)', color = color_palette[1])

ax[1].plot(ts_plot,P_sim[:,0],color = 'k', alpha = 0.5, label = 'P\'(t)')
ax[1].plot(ts_plot,P_sim[:,1:],color = 'k', alpha = 0.5)
ax[1].plot(ts_plot,P_proccessed.detach().numpy(), label = 'P(t)', color = color_palette[2])
ax[0].set_ylabel('E [events per 20 kyr]', fontsize=fontsize)
ax[1].set_ylabel('P', fontsize=fontsize)

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


x = 5_000
ax[0].text(x,14.4, '(a)', 
                fontsize=fontsize)
ax[1].text(x,0.92, '(b)',
                fontsize=fontsize)
ax[2].text(x,2.85, '(c)',
                fontsize=fontsize)
ax[3].text(x,2.1, '(d)',
                fontsize=fontsize)

ax[3].set_xlabel('Time [ka b2k]')
plt.savefig('Figures/Fit_alpha_20_gamma_20_sigma_2_dpi_300.png', bbox_inches='tight', dpi = 300)
plt.show()