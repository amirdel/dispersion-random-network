# Copyright 2017 Amir Hossein Delgoshaie, amirdel@stanford.edu
#
# Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee
# is hereby granted, provided that the above copyright notice and this permission notice appear in all
# copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE
# INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE
# FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
# ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import numpy as np
import os
import time
import matplotlib.pyplot as plt
from py_dp.dispersion.transition_matrix_fcns import normalize_columns
from py_dp.dispersion.dispersion_visualization_tools import compare_trans_mat, compare_trans_mat_vtheta
from py_dp.visualization.plot_sidebyside_histogram import plot_sidebyside_transition_prob

network_paper_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
matrix_pics_folder = os.path.join(network_paper_folder, 'plots')
if not os.path.exists(matrix_pics_folder):
    os.mkdir(matrix_pics_folder)
matrix_data_folder = os.path.join(network_paper_folder, 'data', 'matrix_data')
t_start = time.time()
# plot specifications
plt.rcParams.update({'font.size': 20})
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rcParams.update({'figure.autolayout': True})
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Stix']})
legend_size = 13
datalabel = r"$data$"
save_name = 'gauss01'
y_correction = 0.0
lw = 1
mat_fmt = 'pdf'

coeff = 10.0
coeff_str = str(coeff).split('.')[0]
print coeff
print '------------------------------------------------------------------------'
print 'plotting transition matrices'
# plot the aggregate transition matrix and verify Chapman-Kolmogorov relation
lag_array = [1,5]
# load the matrics
v_file = os.path.join(matrix_data_folder, 'v_list_with_freq.npy')
v_mat_list = np.load(v_file)
theta_file = os.path.join(matrix_data_folder, 'theta_list_with_freq.npy')
theta_mat_list = np.load(theta_file)
v_file = os.path.join(matrix_data_folder, 'v_list_no_freq.npy')
v_mat_list_nofreq = np.load(v_file)
theta_file = os.path.join(matrix_data_folder, 'theta_list_no_freq.npy')
theta_mat_list_nofreq = np.load(theta_file)
# plot results
# making sure that sum of the columns of the transitions matrices is one
lag = lag_array[1]
# normalize the matrices before plotting
stencil_model = normalize_columns(v_mat_list[0])
stencil_data = normalize_columns(v_mat_list[1])
extended_model = normalize_columns(v_mat_list_nofreq[0])
extended_data = normalize_columns(v_mat_list_nofreq[1])
prefix = 'v'
col_array = [2, 98]
save_name_prefix = 'dt_'+coeff_str+'_'
plot_sidebyside_transition_prob(stencil_model, stencil_data, extended_model, extended_data, lag,
                                prefix, save_name_prefix, matrix_pics_folder, col_array, fmt=mat_fmt)
# the angle plot
stencil_model = normalize_columns(theta_mat_list[0])
stencil_data = normalize_columns(theta_mat_list[1])
extended_model = normalize_columns(theta_mat_list_nofreq[0])
extended_data = normalize_columns(theta_mat_list_nofreq[1])
prefix = 'theta'
col_array = [78]
save_name_prefix = 'theta_dt_' + coeff_str + '_'
plot_sidebyside_transition_prob(stencil_model, stencil_data, extended_model, extended_data, lag,
                                prefix, save_name_prefix, matrix_pics_folder, col_array, fmt=mat_fmt)
# making sure that sum of the columns of the transitions matrices is one
for v_list, theta_list, fstr in zip([v_mat_list, v_mat_list_nofreq],
                                                    [theta_mat_list, theta_mat_list_nofreq],
                                                    ['stencil', 'extended']):
    trans_matrix_v1 = normalize_columns(v_list[0])
    trans_matrix_v2 = normalize_columns(v_list[1])
    trans_matrix_t1 = normalize_columns(theta_list[0])
    trans_matrix_t2 = normalize_columns(theta_list[1])
    # column-wise comparison of the aggregate transition probabilities
    v_str = 'v_mat'+'_'+fstr+'_'+coeff_str
    theta_str = 'theta_mat'+'_'+fstr+'_'+coeff_str
    both_str = 'both_mat'+'_'+fstr+'_'+coeff_str
    fontsize = 14
    plt.rcParams.update({'font.size':fontsize})
    plt.rc('xtick', labelsize=fontsize * 0.8)
    plt.rc('ytick', labelsize=fontsize * 0.8)
    compare_trans_mat(trans_matrix_v1, trans_matrix_v2, lag, matrix_pics_folder, v_str, fmt=mat_fmt)
    compare_trans_mat(trans_matrix_t1, trans_matrix_t2, lag, matrix_pics_folder, theta_str, fmt=mat_fmt)
    compare_trans_mat_vtheta(trans_matrix_v1, trans_matrix_t1, matrix_pics_folder, both_str, fmt=mat_fmt)
print '------------------------------------------------------------'
t_finish = time.time()
print 'Total time: {:.2f} seconds'.format(t_finish-t_start)


