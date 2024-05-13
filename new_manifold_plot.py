


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


b0, q0, q1, tau, Bbar, eta, gammabar = 0.625, -9., 12., 0.902, 3.4164, 0.5, 1.22

gamma = 1.2

N = 10_000
b_null = np.linspace(0.6, 1.58, N)
B_null = - abs(q0 + q1*(b_null - b0))*(b_null - b0)


discont_b = -q0/q1 + b0

localminimum_b = (2*q1 *b0 - q0)/(2*q1)
localminimum_B = - abs(q0 + q1*(localminimum_b - b0))*(localminimum_b - b0)



b = 1.2
A = np.array([[2*q1*(b0 - b)-q0, 1 ], [-1, 0]])  # larger then discont_b

b = 1.000125125125125
A = np.array([[-1*(2*q1*(b0 - b)-q0), 1 ], [-1, 0]])  # smaller then discont_b


# increase the size of the figure
plt.rcParams['figure.figsize'] = [10, 10]

font_size = 25

#get the colors from the seaborn color palette colorblind
colorblind = sns.color_palette("colorblind")

color_slow_manifold = 'black'
color_bifurcation = colorblind[2]             # muted green as in the paper
color_stable_fixpoint_above = colorblind[1]   # orange
color_stable_fixpoint_below = colorblind[-1]  # light blue
color_Oscillating_fixpoint = colorblind[4]       # purple
color_meta_stable_fixpoint = colorblind[7]    # brown

# make a list of all the eigenvalues from discont_b to 2
N = 100000

b_1 = np.linspace(discont_b, np.max(b_null), N)
eigenvalues_1 = np.zeros((N,2), dtype = complex)
for i in range(N):
    A = np.array([[2*q1*(b0 - b_1[i])-q0, 1/tau ], [-1, 0]])
    eigenvalues_1[i,:] = np.linalg.eig(A)[0]


b = discont_b
A = np.array([[0, 1 ], [-1, 0]])  # b equal to discont_b
eig_discont_b = np.linalg.eig(A)[0]


# make a list of all the eigenvalues from 0.5 to discont_b
b_2 = np.linspace(np.min(b_null), discont_b, N)
eigenvalues_2 = np.zeros((N,2), dtype=complex)
for i in range(N):
    A = np.array([[-1*(2*q1*(b0 - b_2[i])-q0), 1/tau ], [-1, 0]])
    eigenvalues_2[i,:] = np.linalg.eig(A)[0]

b_tot = np.concatenate((b_2, np.array([b]), b_1))
eigenvalues_tot = np.concatenate((eigenvalues_2, np.array([eig_discont_b]), eigenvalues_1))


# set fig_stability and fig_manifold side by side with seaborn
sns.set_style("whitegrid")
sns.set_context("paper")
sns.set_style("ticks")

N = 10_000
b_null = np.linspace(0.6, 1.58, N)

# increase the size of the figure
# make the left figure twice as big as the right one
fig, (ax_manifold_1, ax_manifold_2, ax_manifold_3)= plt.subplots(3, 1,
    sharex=True, figsize=(10, 20))

# decrease the gap between the two figures
fig.subplots_adjust(wspace=0.005,hspace = 0.1)

# increase all the text size
plt.rcParams.update({'font.size': 20, 'xtick.labelsize': 20, 
                'ytick.labelsize': 20, 'axes.labelsize': 20})
            

#get the colors from the seaborn color palette colorblind
colorblind = sns.color_palette("colorblind")

color_slow_manifold = 'black'
color_bifurcation = colorblind[2]             # muted green as in the paper
color_stable_fixpoint_above = colorblind[1]   # orange
color_stable_fixpoint_below = colorblind[-1]  # light blue
color_Oscillating_fixpoint = colorblind[4]       # purple



def add_fixpoint_and_nullclines(gamma, alpha = 0, color = 'k'):
    B_null2 = np.linspace(np.min(B_null), np.max(B_null), N)
    b_null2 = gamma +0 * B_null2
    ax.plot( B_null2, b_null2, color = color, linewidth=3)
    
    intersect_index = np.argmin(np.abs(b_null2 - b_null))
    ax.plot( B_null[intersect_index], b_null[intersect_index], 'o', 
                label=r'Fixpoint for $\gamma = $' + str(gamma),
                color = color, markersize=15, markeredgecolor='black', zorder = 10)



ax = ax_manifold_1



ax.set_ylabel(r'$\Delta$ b', fontsize = font_size)

ax.tick_params(axis='y', which='major', labelsize=25)


# plot the lower stable manifold
cuttoff_lower = np.argmin(np.abs(b_null - localminimum_b))
ax.plot( B_null[:cuttoff_lower], b_null[:cuttoff_lower], color=color_stable_fixpoint_below,
                linewidth=5, label = 'Lower stable')

# plot the upper stable manifold
cuttoff_upper = np.argmin(np.abs(b_null - discont_b))
ax.plot( B_null[cuttoff_upper:], b_null[cuttoff_upper:], color=color_stable_fixpoint_above,
                linewidth=5, label = 'Upper stable')

# plot the Oscillating manifold
ax.plot( B_null[cuttoff_lower:cuttoff_upper], b_null[cuttoff_lower:cuttoff_upper], color=color_Oscillating_fixpoint,
        linestyle='--', linewidth=5, label = 'Oscillating')

# make the bifurcation points a larger circle with a black outline
ax.plot( 0, discont_b,'o', color = color_bifurcation, markersize=15, 
                            markeredgecolor='black', zorder = 10)
ax.plot( localminimum_B, localminimum_b, 'o', label= 'Bifuraction point',
                            color = color_bifurcation, markersize=15,
                            markeredgecolor='black', zorder = 10)
print('lower bifurcation point: B = ', localminimum_B, '; b = ', localminimum_b)


add_fixpoint_and_nullclines(1.5, color = color_stable_fixpoint_above)
handles1, labels1 = ax.get_legend_handles_labels()

ax.text(-3,1.6, '(a)', fontsize = font_size)

ax = ax_manifold_2

ax.set_ylabel(r'$\Delta$ b', fontsize = font_size)

ax.tick_params(axis='both', which='major', labelsize=15)



# plot the lower stable manifold
cuttoff_lower = np.argmin(np.abs(b_null - localminimum_b))
ax.plot( B_null[:cuttoff_lower], b_null[:cuttoff_lower], color=color_stable_fixpoint_below,
                linewidth=5)

# plot the upper stable manifold
cuttoff_upper = np.argmin(np.abs(b_null - discont_b))
ax.plot( B_null[cuttoff_upper:], b_null[cuttoff_upper:], color=color_stable_fixpoint_above,
                linewidth=5)

# plot the Oscillating manifold
ax.plot( B_null[cuttoff_lower:cuttoff_upper], b_null[cuttoff_lower:cuttoff_upper], color=color_Oscillating_fixpoint,
        linestyle='--', linewidth=5)

# make the bifurcation points a larger circle with a black outline
ax.plot( 0, discont_b,'o', color = color_bifurcation, markersize=15, 
                            markeredgecolor='black', zorder = 10)
ax.plot( localminimum_B, localminimum_b, 'o',
                            color = color_bifurcation, markersize=15,
                            markeredgecolor='black', zorder = 10)
print('lower bifurcation point: B = ', localminimum_B, '; b = ', localminimum_b)



# find mid point between the two bifurcation points in b
midpoint_b = (discont_b + localminimum_b)/2
add_fixpoint_and_nullclines(1.2, color = color_Oscillating_fixpoint)




plt.ylabel('b')

# deinge variables for arrows
arrow_head_width = 0.05
arrow_head_length = 0.05

## add a shadow to the manifold from discontinuity to bifurcation point
# get the points on the line
shadow_dits_1 = 0.25
b_shadow_upper = np.linspace(discont_b +0.01, 1.558, N)
B_shadow_upper = - abs(q0 + q1*(b_shadow_upper - b0))*(b_shadow_upper - b0) + shadow_dits_1
ax.plot( B_shadow_upper, b_shadow_upper
            , color=color_Oscillating_fixpoint, linewidth=4, label = 'Limit cycle')
# add an arrow from the end of the shadow to the slow manifold below
ax.arrow( B_shadow_upper[0], b_shadow_upper[0], 0, -0.7,
            head_width= arrow_head_length, head_length=arrow_head_width,
            fc=color_Oscillating_fixpoint, ec=color_Oscillating_fixpoint,
            width=0.02)

## add a shadow below the lower stable manifold from discontinuity to bifurcation point
# get the points on the line
b_shadow_lower = np.linspace(np.min(b_null)-0.005, 1 - 0.01, N)
B_shadow_lower = - abs(q0 + q1*(b_shadow_lower - b0))*(b_shadow_lower - b0) - shadow_dits_1+0.12
ax.plot( B_shadow_lower, b_shadow_lower
            , color=color_Oscillating_fixpoint, linewidth=4)
# add an arrow from the end of the shadow to the slow manifold above
ax.arrow( B_shadow_lower[-1], b_shadow_lower[-1], 0, 0.48,
            head_width= arrow_head_length, head_length=arrow_head_width,
            fc=color_Oscillating_fixpoint, ec=color_Oscillating_fixpoint,
            width=0.015)

handles2, labels2 = ax.get_legend_handles_labels()

ax.text(-3,1.6, '(b)', fontsize = font_size)
ax.tick_params(axis='y', which='major', labelsize=25)


ax = ax_manifold_3

# plot the lower stable manifold
cuttoff_lower = np.argmin(np.abs(b_null - localminimum_b))
ax.plot( B_null[:cuttoff_lower], b_null[:cuttoff_lower], color=color_stable_fixpoint_below,
                linewidth=5)

# plot the upper stable manifold
cuttoff_upper = np.argmin(np.abs(b_null - discont_b))
ax.plot( B_null[cuttoff_upper:], b_null[cuttoff_upper:], color=color_stable_fixpoint_above,
                linewidth=5)

# plot the Oscillating manifold
ax.plot( B_null[cuttoff_lower:cuttoff_upper], b_null[cuttoff_lower:cuttoff_upper], color=color_Oscillating_fixpoint,
        linestyle='--', linewidth=5)

# make the bifurcation points a larger circle with a black outline
ax.plot( 0, discont_b,'o', color = color_bifurcation, markersize=15, 
                            markeredgecolor='black', zorder = 10)
ax.plot( localminimum_B, localminimum_b, 'o', 
                            color = color_bifurcation, markersize=15,
                            markeredgecolor='black', zorder = 10)
print('lower bifurcation point: B = ', localminimum_B, '; b = ', localminimum_b)



# find mid point between the two bifurcation points in b
midpoint_b = (discont_b + localminimum_b)/2

add_fixpoint_and_nullclines(0.7, color = color_stable_fixpoint_below)


### ADD TEXT ###
# add the text fast dynamics using seaborn


ax.set_xlabel('B', fontsize = font_size)
ax.set_ylabel(r'$\Delta$ b', fontsize = font_size)

ax.tick_params(axis='both', which='major', labelsize=25)


# add a,b,c to the subplots
ax.text(-3,1.6, '(c)', fontsize = font_size)





# make a legend for both plots

handles3, labels3 = ax.get_legend_handles_labels()

handles = handles1 + handles2 + handles3
labels  = labels1  + labels2  + labels3

handles_tot = handles
labels_tot = labels

# make a legend in 4 columns below the plot

handles_final = [handles_tot[1], handles_tot[2], handles_tot[0], handles_tot[6], 
                handles_tot[4], handles_tot[5], handles_tot[7], handles_tot[3]]
                
labels_final = [labels_tot[1], labels_tot[2], labels_tot[0], labels_tot[6], 
                labels_tot[4], labels_tot[5], labels_tot[7], labels_tot[3]]

ax.legend(handles_final, labels_final, loc='upper center', ncol=2, fontsize = font_size,
                bbox_to_anchor = (0.45,-.12))
#ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.09), ncol=1, fontsize=20)


plt.savefig('Figures/simple_simple_model_diagram.pdf', bbox_inches='tight')
