import numpy as np 
import matplotlib.pyplot as plt
import sys
import torch
import seaborn as sns
plt.rcParams['text.latex.preamble']= r"\usepackage{wasysym}"

colours = sns.color_palette("colorblind", 3)


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

# Load data
true_loh = np.load('Data/NGRIP_lohman_E_P.npy')
d18O = np.genfromtxt('Data/NGRIP_d18O_all_clean.txt', 
                        skip_header=70, usecols =[0,2])
is_stadial = np.load('Data/stadial_ngrip.npy')
is_stadial = torch.from_numpy(is_stadial).float()

flag = 0
do_events = []

for i, j in enumerate(is_stadial):
    if j != flag:
        do_events.append(i)
        flag = j

do_events = np.array(do_events)[1:].reshape(-1,2)+11700

precursor_list = [104377,84957,59297,58157,54897]
precursor_names = ['23.2', '21.2', '17.2', '16.2', '15.1'][::-1]
fontsize = 20
fig,ax = plt.subplots(2,1, sharex=True, figsize=(20,10))

# reduce the space between the subplots and remove the top line of the bottom plot and vice versa
plt.subplots_adjust(hspace=0)
ax[0].spines['bottom'].set_visible(False)
ax[1].spines['top'].set_visible(False)
ax12 = ax[1].twinx()
ax12.spines['top'].set_visible(False)
ax[1].invert_xaxis()
ax[1].set_xlim(121_000, 11_000)

# set the ax[0] ticks to -35, -40, -45
ax[0].set_yticks([-35,-40,-45])
ax[1].set_xticks([121_000,111_000,101_000,91_000,81_000,
                71_000,61_000,51_000,41_000,31_000,21_000,11_000])
ax[1].set_xticklabels([121,111,101,91,81,71,61,51,41,31,21,11])

ax[1].set_xlabel('Time [ka b2k]', fontsize=fontsize)

ax[0].plot(d18O[:,0], d18O[:,1], label = 'NGRIP', color = colours[0])
ax[0].set_ylabel(r'$\delta^{18}$O [‰]', fontsize=fontsize)

ax[0].set_ylim(-48,-32)
counter_for_precursor = 0
for event in do_events[:-1]:
    # color the area between the event[0] and event[1] grey
    if event[0] in precursor_list:
        print(event, event[1]-event[0])
        if counter_for_precursor == 0:
            ax[0].fill_between((event),-50, -30, color='red', alpha=0.5, label = 'Precursor events', edgecolor = 'face')
        else:
            ax[0].fill_between((event),-50, -30, color='red', alpha=0.5, edgecolor = 'face')
        
        # add the precursor names
        if counter_for_precursor >= 2:
            print(event[0]+3300, precursor_names[counter_for_precursor])
            ax[0].text(event[0]+3350, -33, precursor_names[counter_for_precursor], fontsize = 14, color = 'red')
        else:
            ax[0].text(event[0]-150, -33., precursor_names[counter_for_precursor], fontsize = 14, color = 'red')
        counter_for_precursor += 1

    else:
        ax[0].fill_between((event),-50, -35, color='grey', alpha=0.5, edgecolor = 'face')
ax[0].fill_between((do_events[-1]),-50, -35, color='grey', 
                            alpha=0.5, label = 'Interstadials', edgecolor = 'face')



ax[1].plot(true_loh[0], true_loh[1], label = 'E(t)', color = colours[1])
ax12.plot(true_loh[2], true_loh[3], 
                    label = 'P(t)', color = colours[2])
ax[1].set_ylabel('E(t) [events per 20 kyr]', fontsize=fontsize)
ax12.set_ylabel('P(t) [fraction]', fontsize=fontsize)

# make a combined legend for both plots
lines, labels = ax[0].get_legend_handles_labels()
lines2, labels2 = ax[1].get_legend_handles_labels()
lines3, labels3 = ax12.get_legend_handles_labels()
plt.legend(lines + lines2 + lines3, labels + labels2 + labels3, 
                            loc=[0.02,0.4], fontsize=fontsize)

ax[0].text(10500,-33, '(a)', fontsize = fontsize)
ax[1].text(10500, 12, '(b)', fontsize = fontsize)

# increase the fontsize of the ticks
ax[0].tick_params(axis='both', which='major', labelsize=fontsize)
ax[1].tick_params(axis='both', which='major', labelsize=fontsize)
ax12.tick_params(axis='both', which='major', labelsize=fontsize)
# plt.show()
plt.savefig('Figures/f01_test.pdf', bbox_inches='tight')
