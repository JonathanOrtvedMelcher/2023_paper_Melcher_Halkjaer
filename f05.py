import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Setup the model parameters and make the B nullcline and find the local minimums

b0, q0, q1, tau, Bbar, eta, gammabar = 0.625, -9., 12., 0.902, 3.4164, 0.5, 1.22

N = 10_000
b_null = np.linspace(0.6, 1.58, N)
B_null = - abs(q0 + q1*(b_null - b0))*(b_null - b0)


discont_b = -q0/q1 + b0

localminimum_b = (2*q1 *b0 - q0)/(2*q1)
localminimum_B = - abs(q0 + q1*(localminimum_b - b0))*(localminimum_b - b0)



# increase the size of the figure
plt.rcParams['figure.figsize'] = [10, 10]

#get the colors from the seaborn color palette colorblind
colorblind = sns.color_palette("colorblind")

color_slow_manifold = 'black'
color_bifurcation = colorblind[2]             # muted green as in the paper
color_stable_fixpoint_above = colorblind[1]   # orange
color_stable_fixpoint_below = colorblind[-1]  # light blue
color_Oscillating_fixpoint = colorblind[4]       # purple
color_meta_stable_fixpoint = colorblind[7]    # brown

eigenvalues_color_palette = sns.color_palette("Paired")

color_imag_1 = eigenvalues_color_palette[8]
color_imag_2 = eigenvalues_color_palette[9]

color_real_1 = eigenvalues_color_palette[4]
color_real_2 = eigenvalues_color_palette[5]



def eigenvalues_plot(alpha,ax, N = 10000):
    

    b_1 = np.linspace(discont_b, np.max(b_null), N)
    eigenvalues_1 = np.zeros((N,2), dtype = complex)
    for i in range(N):
        A = np.array([[2*q1*(b0 - b_1[i])-q0, 1/tau ], [-1, alpha/tau]])
        eigenvalues_1[i,:] = np.linalg.eig(A)[0]


    b = discont_b
    A = np.array([[0, 1/tau ], [-1, alpha/tau]])  # b equal to discont_b
    eig_discont_b = np.linalg.eig(A)[0]


    # make a list of all the eigenvalues from 0.5 to discont_b
    b_2 = np.linspace(np.min(b_null), discont_b, N)
    eigenvalues_2 = np.zeros((N,2), dtype=complex)
    for i in range(N):
        A = np.array([[-1*(2*q1*(b0 - b_2[i])-q0), 1/tau ], [-1, alpha/tau]])
        eigenvalues_2[i,:] = np.linalg.eig(A)[0]

    b_tot = (b_2, np.array([b]), b_1)
    eigenvalues_tot = eigenvalues_2, np.array([eig_discont_b]), eigenvalues_1


    # plot the eigenvalues
    for i, b in enumerate(b_tot):
        if i != 1:
            style = '-'

        else:
            style = '.'
        ax.plot(eigenvalues_tot[i][:,0].real, b, style, linewidth=3, label=r'Re $\lambda_1$',color=color_real_1)
        ax.plot(eigenvalues_tot[i][:,1].real, b, style, linewidth=3, label=r'Re $\lambda_2$',color=color_real_2)
        ax.plot(eigenvalues_tot[i][:,0].imag, b, style, linewidth=3, label=r'Im $\lambda_1$',color=color_imag_1)
        ax.plot(eigenvalues_tot[i][:,1].imag, b, style, linewidth=3, label=r'Im $\lambda_2$',color=color_imag_2)
    
    return ax, eigenvalues_tot, b_tot


def jacobian_eig(alpha, b):
    if b > discont_b:
        A = np.array([[2*q1*(b0 - b)-q0, 1/tau ], [-1, alpha/tau]])
    
    if b == discont_b:
        A = np.array([[0, 1/tau ], [-1, alpha/tau]])
    
    if b < discont_b:
        A = np.array([[-1*(2*q1*(b0 - b)-q0), 1/tau ], [-1, alpha/tau]])
    return np.linalg.eig(A)[0]


def tot_eig(alpha, N_1 = 10_000):
    b_1 = np.linspace(discont_b, np.max(b_null), N_1)

    A = np.dstack((np.array((2*q1*(b0 - b_1)-q0,  1/tau*np.ones((N_1)))).T, 
                   np.array((-1 * np.ones((N_1)), alpha/tau*np.ones((N_1)))).T))
    eigenvalues_1 = np.linalg.eig(A)[0]
    
    b = discont_b
    A = np.array([[0, 1/tau ], [-1, alpha/tau]])  # b equal to discont_b
    eig_discont_b = jacobian_eig(alpha, b)


    # make a list of all the eigenvalues from 0.5 to discont_b
    b_2 = np.linspace(np.min(b_null), discont_b, N_1)
    eigenvalues_2 = np.zeros((N_1,2), dtype=complex)

    A = np.dstack((np.array((-1*(2*q1*(b0 - b_2)-q0), 1/tau*np.ones((N_1)))).T,
                     np.array((-1 * np.ones((N_1)), alpha/tau*np.ones((N_1)))).T))
    eigenvalues_2 = np.linalg.eig(A)[0]
    
    b_tot = np.concatenate((b_2, np.array([b]), b_1))
    eigenvalues_tot = np.concatenate((eigenvalues_2, np.array([eig_discont_b]), eigenvalues_1))
    return b_tot, eigenvalues_tot

def stability_check(alpha,b):
    eig = jacobian_eig(alpha, b)
    if np.real(eig[0]) > 0 or np.real(eig[1]) > 0:
        return 2
    
    if np.abs(eig[0].real) < 1e-3 and np.abs(eig[1].real) < 1e-3:
        return 1
    
    return 0

def analytical_anal(alpha, N_1 = 100_000, verbose = False):

    b_tot, eigenvalues_tot = tot_eig(alpha, N_1 = N_1)
    # find the intervals the one real part of the eig values are >0

    b_Oscillating_0 = b_tot[np.real(eigenvalues_tot[:,0]) > 0]
    b_Oscillating_1 = b_tot[np.real(eigenvalues_tot[:,1]) > 0]
    if verbose:
        print('b_tot: ', b_tot)
        print('eigenvalues_tot: ', eigenvalues_tot)
        print('b_Oscillating_0: ', b_Oscillating_0)
        print('b_Oscillating_1: ', b_Oscillating_1)
    if len(b_Oscillating_0) == 0:
        start_Oscillating = np.min(b_Oscillating_1)
        end_Oscillating = np.max(b_Oscillating_1)

    elif len(b_Oscillating_1) == 0:
        start_Oscillating = np.min(b_Oscillating_0)
        end_Oscillating = np.max(b_Oscillating_0)

    else:
        if alpha <= 0:
            start_Oscillating = np.min((np.min(b_Oscillating_0), np.min(b_Oscillating_1)))
            end_Oscillating = np.max((np.max(b_Oscillating_0), np.max(b_Oscillating_1)))

        else:
            b_Oscillating_0 = b_tot[np.real(eigenvalues_tot[:,0]) < 0]
            b_Oscillating_1 = b_tot[np.real(eigenvalues_tot[:,1]) < 0]
            start_Oscillating = b_Oscillating_0
            end_Oscillating =  b_tot[(np.real(eigenvalues_tot[:,1]) < 0) & (np.real(eigenvalues_tot[:,0]) < 0)]

    mask_real_1 = np.abs(np.real(eigenvalues_tot[:,0])) < 1e-3


    # test if the discontinuity point is a bifurcation point
    is_discont_b_bif = np.abs(np.real(jacobian_eig(alpha, discont_b)[0])) < 1e-30


    # if is_discont_b_bif is True: remove them from the mean calculation
    if is_discont_b_bif:
        bifurcation_points_1_b = b_tot[mask_real_1]
        
        bifurcation_points_1_b = np.mean(bifurcation_points_1_b[bifurcation_points_1_b<discont_b])
        bifurcation_points_1_b = np.append(bifurcation_points_1_b,discont_b)

    else:
        if verbose:
            print(b_tot[ mask_real_1])
        bifurcation_points_1_b = np.mean(b_tot[ mask_real_1])
    
    # make bifurcation_points_1_b a numpy array

    if type(bifurcation_points_1_b) != type(np.array([1,2])):

        bifurcation_points_1_b = np.array([bifurcation_points_1_b])
    # the first is the beginning of the Oscillating part and the second is the end
    # this is the bifurcation point if nan then there is no bifurcation point
    if verbose:
        if np.isnan(bifurcation_points_1_b).any():
            print('no bifurcation point')
        
        else:
            for point in bifurcation_points_1_b:
                print('bifurcation point at: ', point)
                print('has eigenvalues: ', jacobian_eig(alpha, point))


    return start_Oscillating, end_Oscillating, bifurcation_points_1_b


def add_fixpoint_and_nullclines(gamma,ax, alpha = 0, 
    verbose = False, N_1 = 100_000, linewidth = 3, b_start = 0, b_end = 10):

    b_null_true = np.linspace(0.6, 1.58, N_1)
    B_null_true = - abs(q0 + q1*(b_null_true - b0))*(b_null_true - b0)

    B_null_plot = np.linspace(np.min(B_null_true), np.max(B_null_true), N_1)
    b_null_plot = gamma - alpha * B_null_plot

    b_null = np.linspace(b_start, b_end, N)
    B_null = - abs(q0 + q1*(b_null - b0))*(b_null - b0)


    intersections = np.argwhere(np.diff(np.sign(B_null + (b_null-gamma)/alpha))).flatten()

    stab_of_points = []

    for point in intersections:
        color_point = color_Oscillating_fixpoint   
        label = 'Oscillating fixpoint'

        stab_int = stability_check(alpha, b_null[point])
        stab_of_points.append(stab_int)
        if verbose:
            stability_names = ['stabel', 'bifurcation', 'Oscillating']
            print('\n')
            print('Intersection at (', B_null[point],',', b_null[point],')')
            print('with eigenvalues: ', jacobian_eig(alpha, b_null[point]), stability_names[stab_int])
            
        
        if stab_int == 0:
            if b_null[point] > discont_b:
                color_point = color_stable_fixpoint_above
                label = 'Interstadial fixpoint'
            elif b_null[point] < localminimum_b:
                color_point = color_stable_fixpoint_below
                label = 'Stadial fixpoint'
        elif stab_int == 1:
            color_point = color_bifurcation
            label = 'Bifurcation point'

        ax.plot( B_null[point], b_null[point], 'o', 
        label= label,
        color = color_point, markersize=15, markeredgecolor='black')
    
    linestyle = '--'
    label = 'Oscillating nullcline'
    stab_of_points = np.array(stab_of_points)

    if len(intersections) > 1:
        # check if all intercepts are Oscillating,2, then the whole line is Oscillating
        if np.all(stab_of_points == 2):
            color_line = color_Oscillating_fixpoint
            label = 'Oscillating nullcline'
            linestyle = '-'
        else:
            color_line = 'grey'
            label = 'Multistable intercepts'
        
    else: 
        color_line = color_point
        if color_line == color_bifurcation:
            label = 'Bifurcation nullcline'
        elif color_line == color_stable_fixpoint_above:
            label = 'Interstadial nullcline'
        elif color_line == color_stable_fixpoint_below:
            label = 'Stadial nullcline'
        linestyle = '-'
    

    ax.plot( B_null_plot, b_null_plot, label= label,
    color = color_line, linewidth=linewidth, linestyle = linestyle)


@np.vectorize
def dB_dt(delta_b, B, gamma, alpha):
    return delta_b+ alpha*B - gamma

@np.vectorize
def d_delta_b_dt(delta_b, B, q0, q1, b0):
    return -B - np.abs(q0 + q1 * (delta_b - b0))*(delta_b - b0)


def plot_stab_anal(ax,alpha, gamma, verbose = False):

    if verbose:
        print('='*85)
        print('Analysis of: alpha =', alpha, ' gamma = ', gamma)


    start_Oscillating, end_Oscillating, bifurcation_points = analytical_anal(alpha, verbose = verbose)

    if alpha <= 0:
        # plot the lower stable manifold

        cuttoff_lower = np.argmin(np.abs(b_null - start_Oscillating))
        cuttoff_upper = np.argmin(np.abs(b_null - end_Oscillating))
        ax.plot( B_null[:cuttoff_lower], b_null[:cuttoff_lower], color=color_stable_fixpoint_below,
                        linewidth=5)

        # plot the upper stable manifold
        cuttoff_upper = np.argmin(np.abs(b_null - end_Oscillating))
        ax.plot( B_null[cuttoff_upper:], b_null[cuttoff_upper:], color=color_stable_fixpoint_above,
                        linewidth=5)

        # plot the Oscillating manifold
        ax.plot( B_null[cuttoff_lower:cuttoff_upper], b_null[cuttoff_lower:cuttoff_upper], color=color_Oscillating_fixpoint,
                linestyle='--', linewidth=5)
    
    else:
        cuttoff_lower = np.argmin(np.abs(b_null - np.min(end_Oscillating)))
        cuttoff_upper = np.argmin(np.abs(b_null - np.max(end_Oscillating)))
        ax.plot( B_null[0:cuttoff_lower], b_null[0:cuttoff_lower], color=color_Oscillating_fixpoint,
                linestyle='--', linewidth=5)
    
        ax.plot(B_null[cuttoff_lower:cuttoff_upper], b_null[cuttoff_lower:cuttoff_upper], 
        color=color_stable_fixpoint_below, linestyle='-', linewidth=5)

        ax.plot(B_null[cuttoff_upper:], b_null[cuttoff_upper:], color=color_Oscillating_fixpoint,
                linestyle='--', linewidth=5)

    if verbose:
        print(start_Oscillating, end_Oscillating)
        print('Oscillating range is:', b_null[cuttoff_lower], b_null[cuttoff_upper])



    # make the bifurcation points a larger circle with a black outline

    if not np.isnan(bifurcation_points[0]):    
        for point in bifurcation_points:
            # find the closed B to the bifurcation point
            ind = np.argmin(np.abs(b_null - point))
            B_biff = B_null[ind]
            ax.plot( B_biff, point,'o', color = color_bifurcation, markersize=15, 
                                markeredgecolor='black', label = 'Bifurcation point')



    add_fixpoint_and_nullclines(gamma, ax,alpha, verbose = verbose)


    return b_null[cuttoff_lower:cuttoff_upper][0], b_null[cuttoff_lower:cuttoff_upper][-1]




fig, ax_tot = plt.subplots(3,2, sharey=True, figsize=(25, 20))
# reduce white space between subplots
fig.subplots_adjust(hspace=0.1, wspace=0.1)
gammas_tot = [
        [.4,.8,1.2],
        [0.9,1.2,1.5],
        [2.2,1.6,3.2],
]

colors_tot = [
        [color_stable_fixpoint_below, color_meta_stable_fixpoint, color_stable_fixpoint_above],
        [color_stable_fixpoint_below, color_meta_stable_fixpoint, color_stable_fixpoint_above],
        [color_Oscillating_fixpoint, color_Oscillating_fixpoint, color_Oscillating_fixpoint],
]

alphas_tot = [ 0.6, 0, -1.5]


for ax,gammas,colors, alpha in list(zip(ax_tot,gammas_tot,colors_tot,alphas_tot)):
            
    for gamma in gammas:
        cuttoff_lower, cuttoff_upper = plot_stab_anal(ax[0],alpha,gamma,verbose = False)

    _, eigenvalues_tot, b_tot = eigenvalues_plot(alpha,ax[1])
    ax[1].set_xlim(-15,15)
    eigenvalues_tot = np.concatenate((eigenvalues_tot[0], eigenvalues_tot[1], eigenvalues_tot[2]))
    b_tot = np.concatenate((b_tot[0], b_tot[1], b_tot[2]))
    if alpha <= 0:        
        ax[1].axhspan(cuttoff_lower, cuttoff_upper, alpha=0.2, 
            color=color_Oscillating_fixpoint, label = 'Oscillating')

        ax[1].axhspan(np.min(b_null), cuttoff_lower, alpha=0.2,
            color= color_stable_fixpoint_below, label = 'Stadial')

        ax[1].axhspan(cuttoff_upper, np.max(b_null), alpha=0.2,
            color= color_stable_fixpoint_above, label = 'Interstadial')

        if cuttoff_lower > 0.6:
            ax[1].hlines(cuttoff_lower, -20,20, 
                color= color_bifurcation, label = 'Bifurcation')
    
    else:
        ax[1].axhspan(np.min(b_null),cuttoff_lower, alpha=0.2, 
            color=color_Oscillating_fixpoint, label = 'Oscillating')

        ax[1].axhspan(cuttoff_lower, cuttoff_upper, alpha=0.2,
            color= color_stable_fixpoint_below, label = 'Stadial')

        ax[1].axhspan(cuttoff_upper, np.max(b_null), alpha=0.2,
            color = color_Oscillating_fixpoint, label = 'Oscillating')

        ax[1].hlines(cuttoff_upper, -20,20, 
            color= color_bifurcation, label = 'Bifurcation')

    if cuttoff_upper < 1.55:
        # check if the discont_b is a bifurcation point
        is_discont_b_bif = np.abs(np.real(jacobian_eig(alpha, discont_b)[0])) < 1e-30
        if is_discont_b_bif:
            ax[1].hlines(cuttoff_upper, -20,20,
                color= color_bifurcation, label = 'Bifurcation')



# set x and y limits
ax_tot[0,0].set_ylim(0.5,1.6)
ax_tot[0,0].set_xlim(-2.5,0.5)
for i in range(len(ax_tot)):
    ax_tot[i,0].set_xlim(-2.5,0.5)
    # remove the x labels for all but the bottom plots
    if i != 2:
        ax_tot[i,0].set_xticklabels([])
        ax_tot[i,1].set_xticklabels([])

    else:
        ax_tot[i,0].set_xlabel('B', fontsize=20)
        ax_tot[i,1].set_xlabel('Eigenvalues', fontsize=20)
    ax_tot[i,0].set_ylabel(r'$\Delta b$', fontsize=20)

# make a legend with for the left

handles_tot_0 = []
labels_tot_0 = []
for i in range(len(ax_tot)):
    handles, labels = ax_tot[i,0].get_legend_handles_labels()
    for i, label in enumerate(labels):
        if label not in labels_tot_0:
            handles_tot_0.append(handles[i])
            labels_tot_0.append(label)

handles_tot_0 =[
        handles_tot_0[5], handles_tot_0[6], 
        handles_tot_0[1], handles_tot_0[2], 
        handles_tot_0[3], handles_tot_0[4],
        handles_tot_0[0], handles_tot_0[7],
]

labels_tot_0 =[
        labels_tot_0[5], labels_tot_0[6], 
        labels_tot_0[1], labels_tot_0[2],
        labels_tot_0[3], labels_tot_0[4],
        labels_tot_0[0], labels_tot_0[7], 
]

# make a legend with for the right
handles_tot_1 = []
labels_tot_1 = []
for i in range(len(ax_tot)):
    handles, labels = ax_tot[i,1].get_legend_handles_labels()
    for i, label in enumerate(labels):
        if label not in labels_tot_1:
            handles_tot_1.append(handles[i])
            labels_tot_1.append(label)



ax_tot[-1,0].legend(handles_tot_0, labels_tot_0,
                        loc='upper center', bbox_to_anchor=(0.5, -0.14),ncol=4, fontsize=15)
ax_tot[-1,1].legend(handles_tot_1, labels_tot_1,
                        loc='upper center', bbox_to_anchor=(0.5, -0.14),ncol=4, fontsize=15)

# increase the text size for ticks and labels
for i in range(len(ax_tot)):
    ax_tot[i,0].tick_params(axis='both', which='major', labelsize=20)
    ax_tot[i,1].tick_params(axis='both', which='major', labelsize=20)
    ax_tot[i,0].tick_params(axis='both', which='minor', labelsize=20)
    ax_tot[i,1].tick_params(axis='both', which='minor', labelsize=20)

# add alpha = for each row on the right side and tilted 270 degrees
for i in range(len(ax_tot)-1):
    ax_tot[i,1].text( 15, 0.9, r'$\alpha = $'+str(alphas_tot[i]), fontsize=20, rotation=270)

ax_tot[2,1].text(15, 1, r'$\alpha =$', fontsize=20, rotation=270)

# Add the -0.6 part separately with a larger font size
ax_tot[2,1].text(14.95, 0.94, r'$-$', fontsize=24, rotation=270)
ax_tot[2,1].text(15, .83, r'$0.6$', fontsize=20, rotation=270)

# add alphabetic labels to the subplots 
import string
for ax_sub in ax_tot.flatten():
    x,y = -2.85, 1.55
    if ax_tot.flatten().tolist().index(ax_sub) % 2 == 1:
        x = -16.74
    ax_sub.text(x,y, '(' + string.ascii_lowercase[ax_tot.flatten().tolist().index(ax_sub)] + ')', 
        fontsize=25)


# save the figure
plt.savefig('figures/f05.pdf', bbox_inches='tight')
plt.show()