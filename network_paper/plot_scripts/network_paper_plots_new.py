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
# plt.rc('font',**{'family':'serif','serif':['Caladea']})
plt.rc('font',**{'family':'serif','serif':['Stix']})
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{amssymb}']
legend_size = 13

# three_way = True
# purt = True
# link_length = True
# trajectory = True
# matrix = True
# kde = True

# What to plot
three_way = True
purt = False


link_length = False
trajectory = False
matrix = False
kde = False
plume_fmt = 'pdf'
# model_labels = [r'$(log(v), \theta)$', r'$(log(v), \theta, f)$']
model_labels = [r'$(v, \theta)$', r'$(v, \theta, f)$']
###################################################################################################
if purt:
    stencil_savefig_folder = 'purt'
    #local sample path
    sigma_folder = 'sigma_2'
    multiplier = 20
    stencil_names = ['dt_2dxy_20_realz_400_100_5_100v.pkl', 'dt3dxy_50_realz_25_100_5_100v.pkl']
    datalabel = r"$data$"
    save_base_folder = '/home/amirhossein/Desktop/test_pics2'
    data_save_folder = os.path.join(save_base_folder, stencil_savefig_folder + '_' + 'data')
    main_folder = '/home/amirhossein/research_data/'
    save_name = sigma_folder
    case_name = 'oneMiddle_100'
    theta = np.pi/4
    moments = False
    plumes = False
    bt = True
    two_d = False
    lw = 1
    fmt = plume_fmt
    data_folder = os.path.join(main_folder, sigma_folder)
    case_folder = os.path.join(data_folder, case_name)
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
    #create save folder
    save_folder = os.path.join(save_base_folder, stencil_savefig_folder)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    loader = np.load(os.path.join(data_save_folder, 't_end.npz'))
    t_end = loader['t_end']
    plot_wrapper_with_saved_data(t_end, dt_mean, stencil_dt, data_save_folder, save_folder, save_name, datalabel,
                                 model_labels, l, theta, y_correction, lw, fmt, moments, plumes, bt)
#
# ################################################################################################################
if three_way:
    stencil_savefig_folder = '2d_3d_compare'
    #local sample path
    multiplier = 20
    stencil_names = ['dt_2dxy_20_realz_400_100_5_100v.pkl', 'dt3dxy_50_realz_25_100_5_100v.pkl']
    datalabel = r"$data$"
    save_base_folder = '/home/amirhossein/Desktop/test_pics2'
    # data_save_folder = os.path.join(save_base_folder, stencil_savefig_folder + '_' + 'data')
    # save_base_folder = '/home/amirhossein/research_data/cees_plots/paper_dt_20_plots'
    data_save_folder = '/home/amirhossein/research_data/cees_plots/dt_study_mean/dt_20/plots/00_none/plumes_data'

    save_name = 'sigma_2'
    case_name = 'oneMiddle_100'
    theta = np.pi/4
    moments = True
    plumes = True
    bt = True
    two_d = True
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


    #create save folder
    save_folder = os.path.join(save_base_folder, stencil_savefig_folder)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    plot_wrapper_with_saved_data(t_end, dt_mean, stencil_dt, data_save_folder, save_folder, save_name, datalabel,
                                 model_labels, l, theta, y_correction, lw, fmt, moments, plumes, bt)
##############################################################################################################
if link_length:
    print 'making tube length plot...'
    def tube_length(network):
        tp_adj = network.tp_adj
        x_array = network.pores.x
        y_array = network.pores.y
        x1 = x_array[tp_adj[:,0]]
        y1 = y_array[tp_adj[:,0]]
        x2 = x_array[tp_adj[:,1]]
        y2 = y_array[tp_adj[:,1]]
        tube_l_array = np.sqrt(np.power(x2-x1,2) + np.power(y2-y1,2))
        return tube_l_array

    print 'loading network'
    network_file = '/home/amirhossein/research_data/network_skeleton/500_non-periodic.npy'
    with open(network_file, 'rb') as input:
        base_network = pickle.load(input)
    print 'done'

    kde_list = []
    ind = np.linspace(0,3,200)
    n_networks = 4
    for i in range(n_networks):
        print i
        purturbed_network = purturb_network(base_network, 0.45)
        purtubed_lengths = tube_length(purturbed_network)
        gkde = gaussian_kde(purtubed_lengths, bw_method=0.1)
        ind = np.linspace(0,3,200)
        kdepdf = gkde.evaluate(ind)
        kde_list.append(kdepdf[:])

    n_networks = 4
    fig = plt.figure(figsize=[6,4])
    # fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hold(True)
    c = ['k', 'r', 'g', 'b', 'k']
    st = ['-', '--', '-.', ':']
    for i in range(n_networks):
        kdepdf = kde_list[i]
        # label = r'$network\;'+str(i+1)+'$'
        label = 'network ' + str(i + 1)
        ax.plot(ind, kdepdf/sum(kdepdf), linestyle = st[i], color = c[i],lw = 1, label=label)
    # ax.set_ylabel(r'$PDF(l)$', fontsize=18)
    # ax.set_xlabel(r'$l$', fontsize=18)
    ax.set_ylabel('probability density')
    ax.set_xlabel('link length')

    ax.ticklabel_format(axis='y', format='sci', scilimits=(-2,2))
    # ax.legend(fontsize=14)
    ax.legend(fontsize=legend_size, loc='best')
    ax.set_xbound([0,2.0])

    save_path = '/home/amirhossein/Dropbox/manuscripts/Markov_time/figures/l_distrib.pdf'
    fig.savefig(save_path, format='pdf')
############################################################################################################
if trajectory:
    from matplotlib.patches import Ellipse
    from py_dp.dispersion.convert_to_time_process_with_freq import get_time_dx_array_with_frequency
    print 'make trajectory plot...'
    save_folder = '/home/amirhossein/Dropbox/manuscripts/Markov_time/figures/'
    save_name_1 = os.path.join(save_folder, 'traj1.pdf')

    address = '/home/amirhossein/research_data/sample_path/real_2.pkl'
    with open(address, 'r') as input:
        data = pickle.load(input)

    def expand_dx_freq(dx_array, freq_array):
        dx_out = np.array([])
        for i in range(len(dx_array)):
            freq  = freq_array[i]
            dx = dx_array[i]
            if freq > 1:
                dx_out = np.hstack((dx_out, dx * np.ones(freq)))
            else:
                dx_out = np.hstack((dx_out, dx))
        return dx_out

    dt = 75.0
    particle_number_array = [0, 12]
    fig = plt.figure(figsize=[6,4])
    ax = fig.add_subplot(1,1,1)
    ax.hold = True
    counter = 0
    for particle_number in particle_number_array:
        last_idx = 300
        t_array = data.t_array[particle_number,:last_idx]
        dt_array = np.diff(t_array)
        x_array = data.x_array[particle_number,:last_idx]
        v_array = np.divide(np.diff(x_array), np.diff(t_array))

        dx, freq = get_time_dx_array_with_frequency(dt_array, v_array, dt)
        new_dx = expand_dx_freq(dx, freq)
        x2 = np.hstack((0.0,np.cumsum(new_dx)))
        new_t = dt*np.arange(len(x2))
        style = 'ro-'
        if counter == 0:
            ax.plot(t_array,x_array, 'b', lw = 1.0, label='original')
            ax.plot(new_t, x2, style, markerfacecolor='none', lw=1.2, label='time averaged')
        else:
            ax.plot(t_array,x_array, 'b', lw = 1.0)
            ax.plot(new_t, x2, style, markerfacecolor='none', lw=1.2)
        counter += 1
    ax.ticklabel_format(axis='x', format='sci', scilimits=(-2,2))
    ax.set_xbound([0,1600])
    ax.set_ybound([0,260])
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.legend(loc='best', fontsize=legend_size)
    # ax.legend(loc='best', fontsize=13)
    el = Ellipse((570, 80), 300, 50, facecolor='w', edgecolor='k', alpha=0.5)
    ax.add_artist(el)
    fig.savefig(save_name_1, format='pdf')
##############################################################################################
if matrix:
    print 'trans matrix plots...'
    from py_dp.dispersion.transition_matrix_fcns import normalize_columns
    from py_dp.dispersion.dispersion_visualization_tools import compare_trans_mat, compare_trans_mat_hist, \
        compare_trans_mat_vtheta
    fig_save_root = '/home/amirhossein/Desktop/test_pics2/transmat_pics'
    data_folder = '/home/amirhossein/Desktop/test_pics2/transmat_data'
    v_mat_list = np.load(os.path.join(data_folder,'v_list.npy'))
    theta_mat_list = np.load(os.path.join(data_folder,'theta_list.npy'))
    lag_str = 'lag'
    lag_array = [1, 5]
    for lag in lag_array:
        lag_str = lag_str + '_' + str(lag)
    main_save_folder = os.path.join(fig_save_root, lag_str)
    if not os.path.exists(main_save_folder):
        os.mkdir(main_save_folder)
    figure_save_folder = main_save_folder
    trans_matrix_v1 = normalize_columns(v_mat_list[0])
    trans_matrix_v2 = normalize_columns(v_mat_list[1])
    trans_matrix_t1 = normalize_columns(theta_mat_list[0])
    trans_matrix_t2 = normalize_columns(theta_mat_list[1])
    lag = lag_array[1]
    compare_trans_mat_hist(trans_matrix_v1, trans_matrix_v2, lag, figure_save_folder, 'v', legend_size=14)
    compare_trans_mat_hist(trans_matrix_t1, trans_matrix_t2, lag, figure_save_folder, 'theta', legend_size=14)
    fontsize = 14
    plt.rcParams.update({'font.size':fontsize})
    plt.rc('xtick', labelsize=fontsize * 0.8)
    plt.rc('ytick', labelsize=fontsize * 0.8)
    compare_trans_mat(trans_matrix_v1, trans_matrix_v2, lag, figure_save_folder, 'v')
    compare_trans_mat(trans_matrix_t1, trans_matrix_t2, lag, figure_save_folder, 'theta')
    compare_trans_mat_vtheta(trans_matrix_v1, trans_matrix_t1, figure_save_folder, 'both_matrix')
#################################################################################################
if kde:
    #2d KDE plots
    fontsize = 24
    plt.rcParams.update({'font.size': fontsize})
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)
    sigma_folder = 'sigma_2'
    multiplier = 20
    stencil_names = ['dt3dxy_10_realz_25_100_5_100v.pkl']
    model_labels = [r'$2d \mbox{-} 20\widetilde{\delta t}$']
    stencil_savefig_folder = '2d_compare'
    save_base_folder = '/home/amirhossein/Desktop/test_pics2'
    data_save_folder = os.path.join(save_base_folder, stencil_savefig_folder + '_' + 'data')
    main_folder = '/home/amirhossein/research_data/'
    save_name = sigma_folder
    case_name = 'oneMiddle_100'
    theta = np.pi/4
    moments = False
    plumes = False
    bt = False
    two_d = True
    lw = 1
    fmt = 'png'

    data_folder = os.path.join(main_folder, sigma_folder)
    case_folder = os.path.join(data_folder, case_name)
    #load one data files and find l and y_correction
    with open(os.path.join(case_folder, 'real_0.pkl')) as input:
        data0 = pickle.load(input)
        y_correction = data0.y_array[0,0]
        l = (data0.x_array[0,1] - data0.x_array[0,0])/np.cos(theta)
        print 'l: ', l
        print 'y0: ', y_correction
    #load the stencil and find the used dt
    t_scale_address = os.path.join(case_folder, 'corr_models', 't_scale.pkl')
    with open(t_scale_address, 'rb') as input:
        dt_mean = pickle.load(input)
    stencil_dt = multiplier*dt_mean
    save_folder = os.path.join(save_base_folder, stencil_savefig_folder)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    loader = np.load(os.path.join(data_save_folder, 't_end.npz'))
    t_end = loader['t_end']
    datalabel = r"$data$"
    plot_wrapper_with_saved_data(t_end, dt_mean, stencil_dt, data_save_folder, save_folder, save_name, datalabel,
                                 model_labels, l, theta, y_correction, lw, fmt, moments, plumes, bt, two_d=two_d)
#################################################################################################
# print 'copy all plots to folder...'
os.system('python copy_paper_plots.py')