
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch



fontsize = 25
line_lohman = np.load(r'Data/NGRIP_lohman_E_P.npy')

E_proccessed = torch.tensor(line_lohman[1][::-1][:84000][::20].copy())
P_proccessed = torch.tensor(line_lohman[3][::-1][:84000][::20].copy())

results = np.load('Data/lots_of_points_results.npy') # mu_e, mu_p,std_e,std_p
points  = np.load('Data/lots_of_points_domain.npy')  #sigma, gamma, alpha

# make a gif of gamma and sigma plots with the mask changing

pair_E_P_slope = np.array([results[:,0], results[:,1]])
delta_pair_E_P_slope = np.array([results[:,2], results[:,3]])

# the shape of the arrays are (sigma, gamma, alpha) make mehsgrid of the
# sigma, gamma and alpha values so 
# that the a index of pair_E_P_slope and delta_pair_E_P_slope
# corresponds to the sigma, gamma and alpha values. And can be back transformed
# to the sigma, gamma and alpha values
gamma_, sigma_, alpha_ = points[:,1], points[:,0], points[:,2]



tot_pair_E_P_slope = np.vstack((pair_E_P_slope, 
                delta_pair_E_P_slope, sigma_, gamma_, alpha_)).T

# the E and P lines start 10k years in the glacial period and ends
# 10k years before the end of the glacial period because of the rolling
# window of 20k years and it steps on year at a time
# therefore to take 1 point every 10k years we need to take the points


E_for_plot_lohmann = line_lohman[1][::-1][::10_000]
P_for_plot_lohmann = line_lohman[3][::-1][::10_000]

def make_plot(fig,mask,ind,ax, add_true_values = False, add_colorbar = False,
color_bar_ax = None):
    tot_pair_E_P_slope_post_mask = tot_pair_E_P_slope[mask]
    scatter = ax.scatter(tot_pair_E_P_slope_post_mask[:,1], 
            tot_pair_E_P_slope_post_mask[:,0], 
            c = tot_pair_E_P_slope_post_mask[:,ind],
             cmap = 'rainbow', zorder = 5, s = 50)
    # Add a colorbar
    if add_colorbar:
        if ind == 4:
            colorbar = fig[0][0][0].colorbar(scatter, cax=color_bar_ax, 
                                ticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                                orientation = 'horizontal',
            pad = 100)
            colorbar.set_label(r'$\sigma$')  # Replace with your desired label
            # set colorbar range
            scatter.set_clim(0,0.3)
        
        else:
            print(ind)
            print(fig[0][0][0])
            colorbar = fig[0][0][0].colorbar(scatter, cax=color_bar_ax, 
                                ticks = [0, 0.5, 1, 1.5, 2, 2.5, 3],
                                orientation = 'horizontal'
            )

            scatter.set_clim(0,3)

            colorbar.set_label(r'$\gamma$')  # Replace with your desired label
    
    if add_true_values:
        ax.plot(P_proccessed, E_proccessed, color = 'k', 
        label = '(P(t),E(t)) Lohmann et al.', linestyle = '--', linewidth = 1,
        alpha = 1, zorder = 100)
        color_lohmann_10k = 'k'
        size_lohmann_10k = 10
        ax.plot(P_for_plot_lohmann, E_for_plot_lohmann, 'x', 
        markersize = size_lohmann_10k, color = color_lohmann_10k, 
        label = '(P(t),E(t)) Lohmann et al. 10k', zorder = 100, alpha = 1)
        # add  text to the plot for each point in P_for_plot_lohmann, E_for_plot_lohmann

        for i, E_P in enumerate(zip(E_for_plot_lohmann,P_for_plot_lohmann)):
            E_current, P_current = E_P
            offset_right = 0
            offset_down = 0.2
            if i in [0,5]:
                # move the text a bit to the down and to the right
                offset_right = 0.01
                offset_down = -0.5

            ax.annotate(f'{i*10 + 21}',
            (E_current + offset_right, 
            P_current  + offset_down), 
            color = color_lohmann_10k, fontsize=fontsize, zorder = 100)

    return ax

def make_sidebyside(fig,ii,ax, add_true_values = False, add_colorbar = False, cax = None):
    mask = tot_pair_E_P_slope[:,6] == np.unique(alpha_)[ii]

    # make the plots
    ax[0] = make_plot(fig,mask, 5, ax[0], add_true_values= add_true_values, 
                        add_colorbar = add_colorbar,
                        color_bar_ax = cax[0])
    ax[1] = make_plot(fig,mask, 4, ax[1], add_true_values= add_true_values, 
                        add_colorbar = add_colorbar,
                        color_bar_ax = cax[1])
    # save the plots
    return fig,ax

# make a gif of the plots

fig,ax = plt.subplots(4,2,figsize=(22,25), sharex=True, sharey=True)
# decrease the space between the plots
fig.subplots_adjust(hspace=0.1, wspace=0.1)
# increase the font size
font = {'size'   : 20}

matplotlib.rc('font', **font)# specify the size of each subfigure 
cbar_ax_1 = fig.add_axes([0.13,0.06,.35,0.01])
cbar_ax_2 = fig.add_axes([0.54,0.06,.35,0.01])
tot_cax = [cbar_ax_1, cbar_ax_2]
print(np.unique(alpha_))
print(len(np.unique(alpha_)))
alpha_index_for_sidebyside = np.unique(alpha_, return_index = True)[1]
print(alpha_index_for_sidebyside)
add_colorbar = False
for i, alpha_index_ in enumerate(alpha_index_for_sidebyside):
    if i == len(alpha_index_for_sidebyside)-1:
        add_colorbar = True
    fig = make_sidebyside(fig,alpha_index_,ax[i], add_true_values = True, 
        add_colorbar = add_colorbar,     cax = tot_cax)
    ax[i][0].set_ylabel(r"$\mu'_E$" + ' [events per 20 kyr]', fontsize=fontsize)
    if i == 0:
    # add text on the right side of the plot with the alpha value
        ax[i][1].text(1.02, 0.5, r'$\alpha$ = ' + 
        f'{np.abs(np.round(np.unique(alpha_)[alpha_index_], 2))}',
        horizontalalignment='center', verticalalignment='center', 
        transform=ax[i][1].transAxes, fontsize=fontsize, rotation = 270)
    else:
        # add text on the right side of the plot with the alpha value
        ax[i][1].text(1.02, 0.5, r'$\alpha$ = – ' +
        f'{np.abs(np.round(np.unique(alpha_)[alpha_index_], 2))}',
        horizontalalignment='center', verticalalignment='center', 
        transform=ax[i][1].transAxes, fontsize = fontsize, rotation = 270)


ax[0][1].legend(loc = 'upper right')
ax[-1][0].set_xlabel(r"$\mu'_P$" , fontsize=fontsize)
ax[-1][1].set_xlabel(r"$\mu'_P$" , fontsize=fontsize)

# increase the font size of the x and y labels and the ticks

for i in range(4):
    for j in range(2):
        ax[i][j].tick_params(axis='both', which='major', labelsize=fontsize)
        ax[i][j].tick_params(axis='both', which='minor', labelsize=fontsize)

import string
for ax_sub in ax.flatten():
    x,y = -0.24, 17.25
    if ax.flatten().tolist().index(ax_sub) % 2 == 1:
        x = -0.14
    ax_sub.text(x,y, '(' + \
    string.ascii_lowercase[ax.flatten().tolist().index(ax_sub)] + ')', fontsize=fontsize)

# add text above left column

ax[0][0].title.set_text(r'Coloured with $\gamma$')
ax[0][0].title.set_fontsize(fontsize)
ax[0][1].title.set_text(r'Coloured with $\sigma$')
ax[0][1].title.set_fontsize(fontsize)

plt.savefig("Figures/Gif_now_as_panel.png", bbox_inches = 'tight', dpi = 300)
plt.show()