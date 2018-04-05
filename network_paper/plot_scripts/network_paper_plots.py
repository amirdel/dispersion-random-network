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

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from py_dp.dispersion.plot_wrapper_functions import plot_wrapper_with_saved_data
from py_dp.dispersion.dispersion_aux_classes import purturb_network
from scipy.stats import gaussian_kde


plt.rcParams.update({'font.size': 20})
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rcParams.update({'figure.autolayout': True})
plt.rc('text', usetex=True)
plt.rc('font',**{'family':'serif','serif':['Stix']})
legend_size = 13

three_way = True
purt = True
kde = True

plume_fmt = 'pdf'
model_labels = [r'$(v, \theta)$', r'$(v, \theta, f)$']
network_paper_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plumes_data_folder = os.path.join(network_paper_folder, 'data', 'plumes_data')
temp_plots_folder = os.path.join(network_paper_folder, 'temp_plots')
if not os.path.exists(temp_plots_folder):
    os.mkdir(temp_plots_folder)
###################################################################################################
if purt:
    # plots for the purturbed network case
    sigma_folder = 'sigma_2'
    multiplier = 20
    datalabel = r"$data$"
    # path to saved data
    data_save_folder = os.path.join(plumes_data_folder, 'purt_data')
    # create temp figure save folder
    save_folder = os.path.join(temp_plots_folder, 'purt')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    theta = np.pi/4
    moments = False
    plumes = True
    bt = True
    two_d = False
    lw = 1
    fmt = plume_fmt
    #load one data files and find l and y_correction
    with open(os.path.join(data_save_folder, 'real_0.pkl')) as input:
        data0 = pickle.load(input)
        y_correction = data0.y_array[0,0]
        l = (data0.x_array[0,1] - data0.x_array[0,0])/np.cos(theta)
        print 'l: ', l
        print 'y0: ', y_correction
    #load the stencil and find the used dt
    t_scale_address = os.path.join(data_save_folder, 't_scale.pkl')
    with open(t_scale_address, 'rb') as input:
        dt_mean = pickle.load(input)
    stencil_dt = multiplier*dt_mean
    loader = np.load(os.path.join(data_save_folder, 't_end.npz'))
    t_end = loader['t_end']
    save_name = sigma_folder
    plot_wrapper_with_saved_data(t_end, dt_mean, stencil_dt, data_save_folder, save_folder, save_name, datalabel,
                                 model_labels, l, theta, y_correction, lw, fmt, moments, plumes, bt)
#
# ################################################################################################################
if three_way:
    #local sample path
    sigma_folder = 'sigma_2'
    multiplier = 20
    datalabel = r"$data$"
    data_save_folder = os.path.join(plumes_data_folder, '2d_3d_compare_data')
    theta = np.pi/4
    moments = True
    plumes = True
    bt = True
    two_d = False
    lw = 1
    fmt = plume_fmt
    y_correction = 0.0
    t_file_path = os.path.join(data_save_folder, 'time_file.npz')
    time_file = np.load(t_file_path)
    t_end, dt_mean = time_file['t_end'], time_file['t_scale']
    stencil_dt = multiplier * dt_mean
    network_length_path = os.path.join(data_save_folder, 'network_specs.npz')
    network_length_file = np.load(network_length_path)
    l = network_length_file['l']
    save_folder = os.path.join(temp_plots_folder, '2d_3d_compare')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    save_name = sigma_folder
    plot_wrapper_with_saved_data(t_end, dt_mean, stencil_dt, data_save_folder, save_folder, save_name, datalabel,
                                 model_labels, l, theta, y_correction, lw, fmt, moments, plumes, bt)
#################################################################################################
if kde:
    # 2d KDE plots
    fontsize = 24
    plt.rcParams.update({'font.size': fontsize})
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    sigma_folder = 'sigma_2'
    multiplier = 20
    stencil_names = ['dt3dxy_10_realz_25_100_5_100v.pkl']
    model_labels = [r'$2d \mbox{-} 20\widetilde{\delta t}$']
    stencil_savefig_folder = '2d_compare'
    data_save_folder = os.path.join(plumes_data_folder, 'planar_plume_data')
    main_folder = '/home/amirhossein/research_data/'
    save_name = sigma_folder
    case_name = 'oneMiddle_100'
    theta = np.pi / 4
    moments = False
    plumes = False
    bt = False
    two_d = True
    lw = 1
    fmt = 'png'
    y_correction  = 0.0
    t_file_path = os.path.join(data_save_folder, 'time_file.npz')
    time_file = np.load(t_file_path)
    t_end, dt_mean = time_file['t_end'], time_file['t_scale']
    network_length_path = os.path.join(data_save_folder, 'network_specs.npz')
    network_length_file = np.load(network_length_path)
    l = network_length_file['l']
    data_folder = os.path.join(main_folder, sigma_folder)
    case_folder = os.path.join(data_folder, case_name)
    stencil_dt = multiplier * dt_mean
    save_folder = os.path.join(network_paper_folder, 'temp_plots','planar')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    datalabel = r"$data$"
    plot_wrapper_with_saved_data(t_end, dt_mean, stencil_dt, data_save_folder, save_folder, save_name, datalabel,
                                 model_labels, l, theta, y_correction, lw, fmt, moments, plumes, bt, two_d=two_d)
#################################################################################################
# print 'copy all plots to the /network_paper/plots directory'
os.system('python copy_paper_plots.py')
# generate all matrix plots
os.system('python matrix_plots_network.py')