import matplotlib.pyplot as plt
import numpy as np
import os
from py_dp.dispersion.dispersion_visualization_tools import plume_location_multiple_realizations, plume_location_vector_xy
import pickle


def particle_histogram(time, idx_start, idx_end, bins, folder, prefix, attrib):
    n_realz = idx_end - idx_start + 1
    plume_array = plume_location_multiple_realizations(time, folder, prefix, n_realz, attrib, idx_start)
    n_list = []
    h_list = []
    for plume in plume_array:
        h, out_bins = np.histogram(plume, bins=bins, density=False)
        h_list.append(h)
        n_list.append(len(plume))
    return n_list,h_list

def check_data_convergence_hist(time, data_folder, save_folder, bins, subset_size, n_realz,
                                save_name, prefix='real', attrib='x_array'):
    mid_x = bins[:-1] + np.diff(bins) / 2
    start_idx = np.arange(0, n_realz, subset_size)
    end_idx = np.hstack((start_idx[1:] - 1, n_realz-1))
    h0_list = []
    n0_list = []
    for istart, iend in zip(start_idx, end_idx):
        print 'start, end: ', istart, iend
        n_list,h_list = particle_histogram(time, istart, iend, bins, data_folder, prefix, attrib)
        h0_list.append(h_list[0])
        n0_list.append(n_list[0])
    #plume shape plot
    fig, ax = plt.subplots(1,1)
    h0_cumsum = np.cumsum(h0_list,0, dtype=float)
    n0_cumsum = np.cumsum(n0_list, 0)
    pdf_list = h0_cumsum/n0_cumsum[:, None]
    for i in range(pdf_list.shape[0]):
        label_str = 'n = '+'{:.1e}'.format(n0_cumsum[i])
        ax.plot(mid_x, pdf_list[i,:], label= label_str)
        ax.set_xlabel('x')
    ax.set_xbound([-5,bins[-1]])
    ax.legend()
    save_path = os.path.join(save_folder, save_name + '_plume_convergence.png')
    ax.set_ylabel('particle concentration')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    fig.savefig(save_path, format='png')
    #norm plot
    fig, ax = plt.subplots(1,1)
    norm_list = []
    last_pdf = pdf_list[-1,:]
    for i in range(pdf_list.shape[0]-1):
        norm_list.append(np.linalg.norm(pdf_list[i,:]-last_pdf))
    ax.plot(n0_cumsum[:-1],norm_list)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    ax.set_ylabel('norm (pdf - finest pdf)')
    ax.set_xlabel('number of particles particles')
    save_path = os.path.join(save_folder, save_name +'_plume_norm.png')
    fig.savefig(save_path, format='png')

def check_moment_convergence(data, save_fig_folder, n_split):
    tmax = np.amin(data.t_array[:,-1])
    t_query = tmax/2
    n_tot = data.x_array.shape[0]
    partitions = np.array(np.floor(np.linspace(0, n_tot, n_split))[1:], dtype=int)
    x1, x2, x3 = ([] for i in range(3))
    y1, y2, y3 = ([] for i in range(3))
    for j in partitions:
        print 'partition: ', j
        xout, yout = plume_location_vector_xy(t_query, data.x_array[:j,:], data.y_array[:j,:], data.t_array[:j,:])
        com_x = np.mean(xout)
        com_y = np.mean(yout)
        x1.append(com_x)
        y1.append(com_y)
        msd_x = np.sqrt(np.mean(np.power(xout - com_x,2)))
        msd_y = np.sqrt(np.mean(np.power(yout - com_y,2)))
        x2.append(msd_x)
        y2.append(msd_y)
        kurt_x = np.mean(np.power((xout-com_x)/msd_x,3))
        kurt_y = np.mean(np.power((yout-com_y)/msd_y,3))
        x3.append(kurt_x)
        y3.append(kurt_y)
    x1 = x1/x1[-1]
    x2 = x2/x2[-1]
    x3 = x3/x3[-1]
    y1 = y1/y1[-1]
    y2 = y2/y2[-1]
    y3 = y3/y3[-1]
    #plot x-3
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hold(True)
    ax.plot(partitions, x1, label = r"$\overline{x} / \overline{x}^f$")
    ax.plot(partitions, x2, label = r"$\sigma_x / \sigma_x^f$")
    ax.plot(partitions, x3, label = r"$\gamma_{1x} / \gamma_{1x}^f$")
    ax.legend(loc='best')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    ax.set_xlabel('number of particles')
    fig_path = os.path.join(save_fig_folder, 'x_moments.pdf')
    fig.savefig(fig_path, format='pdf')

    # plot x-2
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hold(True)
    ax.plot(partitions, x1, label=r"$\overline{x} / \overline{x}^f$")
    ax.plot(partitions, x2, label=r"$\sigma_x / \sigma_x^f$")
    ax.legend(loc='best')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax.set_xlabel('number of particles')
    fig_path = os.path.join(save_fig_folder, 'x_moments2.pdf')
    fig.savefig(fig_path, format='pdf')

    #plot y
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hold(True)
    ax.plot(partitions, y1, label = r"$\overline{y} / \overline{y}^f$")
    ax.plot(partitions, y2, label = r"$\sigma_y / \sigma_y^f$")
    ax.plot(partitions, y3, label = r"$\gamma_{1y} / \gamma_{1y}^f$")
    ax.legend(loc='best')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
    ax.set_xlabel('number of particles')
    fig_path = os.path.join(save_fig_folder, 'y_moments.pdf')
    fig.savefig(fig_path, format='pdf')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hold(True)
    ax.plot(partitions, y1, label=r"$\overline{y} / \overline{y}^f$")
    ax.plot(partitions, y2, label=r"$\sigma_y / \sigma_y^f$")
    ax.legend(loc='best')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax.set_xlabel('number of particles')
    fig_path = os.path.join(save_fig_folder, 'y_moments2.pdf')
    fig.savefig(fig_path, format='pdf')