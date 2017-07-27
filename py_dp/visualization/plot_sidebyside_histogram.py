import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import os as os


def plot_sidebyside_transition_prob(stencil_model, stencil_data, extended_model, extended_data, lag,
                                    prefix, save_name_prefix, fig_save_folder, col_array, fmt='pdf'):
    legend_size = 13
    # raise lag one matrices to the power of lag
    stencil_markov = copy(stencil_model)
    for i in range(lag-1):
        stencil_markov = np.dot(stencil_model, stencil_markov)
    extended_markov = copy(extended_model)
    for i in range(lag - 1):
        extended_markov = np.dot(extended_model, extended_markov)
    if prefix.startswith('v'):
        next_str = 'next velocity class'
        label_str1 = r"$T_5^{v}(i,j)$"
        label_str2 = r"${T_1^{v}(i,j)}^5$"
    else:
        next_str = 'next angle class (radians)'
        label_str1 = r"$T_5^{\theta}(i,j)$"
        label_str2 = r"${T_1^{\theta}(i,j)}^5$"
    mat_size = stencil_model.shape[0]
    if prefix == 'theta':
        ax_bound = [-np.pi, np.pi]
        index = np.linspace(-np.pi, np.pi, num=mat_size)
    else:
        ax_bound = [0,mat_size]
        index = np.linspace(0,mat_size,num=mat_size)
    for col in col_array:
        max_y = 1.1*(max(max(stencil_markov[:,col]), max(extended_markov[:,col]),
                         max(stencil_data[:,col]), max(extended_data[:,col])))
        max_y = min(max_y, 1)
        fig = plt.figure(figsize=[12,4])
        ax = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1,2,2)
        # plot the stencil method plots
        ax.step(index, stencil_data[:,col], where='mid', label= label_str1)
        ax.hold(True)
        ax.step(index, stencil_markov[:, col], 'g--',where='mid', label=label_str2)
        # plot the extended stencil plots
        ax2.step(index, extended_data[:,col], where='mid', label= label_str1)
        ax2.hold(True)
        ax2.step(index, extended_markov[:, col], 'g--',where='mid', label=label_str2)
        # plot labels
        ax.set_ylabel("probability")
        ax2.legend(fontsize=legend_size, loc='best')
        for ax_handle in [ax, ax2]:
            ax_handle.set_xlabel(next_str)
            ax_handle.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
            ax_handle.set_ybound([0, max_y])
            ax_handle.set_xbound(ax_bound)
        fig_name = save_name_prefix+'side_hist_'+ str(col).split('.')[0] + '.' + fmt
        file_name = os.path.join(fig_save_folder, fig_name)
        fig.savefig(file_name, format=fmt)