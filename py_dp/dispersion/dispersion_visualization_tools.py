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
import bisect as bs
import pickle
import cPickle as cPickle
import os
from scipy import stats
import matplotlib.pyplot as plt
from copy import copy
from py_dp.dispersion.binning import abs_vel_log_bins_low_high, get_cdf_from_bins
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import gridspec


class dispersionPostProcessModel(object):
    def __init__(self, address, label):
        self.address = address
        self.label = label

class particleTrackData(object):
    def __init__(self, folder_address, n_realization, prefix, label):
        self.folder = folder_address
        self.label = label
        self.n_realization = n_realization
        self.prefix = prefix


class data_holder(object):
    def __init__(self, address, label='data'):
        self.x_array = np.load(os.path.join(address, 'big_x.npy'))
        self.y_array = np.load(os.path.join(address, 'big_y.npy'))
        self.t_array = np.load(os.path.join(address, 'big_t.npy'))
        self.label = label


class model_holder(object):
    def __init__(self, address, label, y_correction=0.0):
        with open(address, 'rb') as input:
            dataHolder = pickle.load(input)
        self.x_array = dataHolder.x_array
        self.y_array = dataHolder.y_array + y_correction
        self.t_array = dataHolder.t_array
        self.label = label


def plume_location_at_given_time(t, x_array, time_array):
    """

    :param t: output time
    :param x_array: location array
    :param time_array: time array
    :return: plume location
    """
    t_max = np.amin(time_array[:,-1])
#     if t>t_max:#not in limits:
#         if (self.n_particles == len(np.where(self.freeze_array)[0])):
#             print "ok since all particles are in the sink..."
#         else:
#             raise Exception("time is larger than simulation time!")
    n_particles = time_array.shape[0]
    return_x = np.zeros(n_particles)
    n_steps = time_array.shape[1]
    assert(n_particles == len(return_x))
    for p in range(n_particles):
        t_temp = time_array[p, :]
        x_temp = x_array[p, :]
        #index of array member to the left
        idx = bs.bisect_left(t_temp, t)
        if idx >= n_steps:
            return_x[p] = x_temp[-1]
        elif idx == 0:
            return_x[p] = x_temp[0]
        else:
            temp1 = t_temp[idx-1:idx+1]
            temp2 = x_temp[idx-1:idx+1]
            return_x[p] = np.interp(t, temp1, temp2)
    return return_x

def plume_location_vector_xy(t, x_array, y_array, time_array):
    t_out = time_array[:,-1] < t
    idx = np.argmax(time_array>t, axis=1)
    idx[idx==0] += 1
    xout = interpolate_arrays(t, idx, x_array, time_array)
    xout[t_out] = x_array[:,-1][t_out]
    yout = interpolate_arrays(t, idx, y_array, time_array)
    yout[t_out] = y_array[:,-1][t_out]
    return xout, yout

def time_to_location_vector(x_target, x_array, time_array, verbose=False):
    n_particles = x_array.shape[0]
    idx = np.argmax(x_array > x_target, axis=1)
    # if idx == 0 that row never got to value x
    # only use indices that got to the x value
    select_idx = np.where(idx>0)[0]
    if verbose:
        print 'percentage arriving at plane: ', len(select_idx)/float(n_particles)
    tout = interpolate_arrays(x_target, idx[select_idx], time_array[select_idx,:], x_array[select_idx,:])
    if len(tout)<n_particles:
        out = np.zeros(n_particles)
        out[:len(tout)] = tout
    else:
        out = tout
    return out

def plume_bt_multiple_locations(x_target, big_x, big_t):
    n_locations = len(x_target)
    n_particles = big_x.shape[0]
    t_out_array = np.zeros((n_locations, n_particles))
    for idx, x in enumerate(x_target):
        t_out_array[idx,:] = time_to_location_vector(x, big_x, big_t)
    return t_out_array

def interpolate_arrays(t, idx, x_array, time_array):
    """
    for each line of x_array, t_array interpolate x using the corresponding index idx
    :param t:
    :param idx:
    :param x_array:
    :param time_array:
    :return:
    """
    x1 = np.array([x_array[i, idx[i] - 1] for i in range(len(idx))])
    x2 = np.array([x_array[i, idx[i]] for i in range(len(idx))])
    t1 = np.array([time_array[i, idx[i] - 1] for i in range(len(idx))])
    t2 = np.array([time_array[i, idx[i]] for i in range(len(idx))])
    ta = t * np.ones(time_array.shape[0])
    return x1 + np.multiply(np.divide(x2-x1, t2-t1), ta - t1)

def plume_location_multiple_times(t_target, big_x, big_y, big_t):
    n_times = len(t_target)
    n_particles = big_x.shape[0]
    x_out_array = np.zeros((n_times, n_particles))
    y_out_array = np.zeros((n_times, n_particles))
    for idx, t in enumerate(t_target):
        x_out_array[idx,:] , y_out_array[idx,:] = plume_location_vector_xy(t, big_x, big_y, big_t)
    return x_out_array, y_out_array

def plume_msd_com_multiple_times(target_time_array, dataHolder):
    plume_x, plume_y = plume_location_multiple_times(target_time_array, dataHolder.x_array,
                                                     dataHolder.y_array, dataHolder.t_array)
    com_x = np.mean(plume_x, 1)
    msd_x = np.sqrt(np.mean(np.power(plume_x - com_x[:,None], 2), 1))
    com_y = np.mean(plume_y, 1)
    msd_y = np.sqrt(np.mean(np.power(plume_y - com_y[:,None], 2), 1))
    return com_x, msd_x, com_y, msd_y



def plume_location_model(t_target_array, address, attrib='x_array'):
    """
    return the plume location for a model at given times
    """
    with open(address, 'rb') as input:
        dataHolder = pickle.load(input)
    x_mat = getattr(dataHolder, attrib)
    t_mat = dataHolder.t_array
    n_given_times = len(t_target_array)
    n_particles = t_mat.shape[0]
    out_put_array = np.zeros((n_given_times, n_particles))
    for i in range(n_given_times):
        t = t_target_array[i]
        out_put_array[i, :] = plume_location_at_given_time(t, x_mat, t_mat)
    return out_put_array

def save_big_data_array(folder, start, n_realization, prefix='real'):
    """
    This function will concatenate all the realization results into 3 files and save the results
    :param folder:
    :param start:
    :param n_realization:
    :param prefix:
    :return:
    """
    file_name = prefix + '_' + str(start) + ".pkl"
    file_address = os.path.join(folder, file_name)
    with open(file_address, 'rb') as input:
        dataHolder = pickle.load(input)
    x_big = dataHolder.x_array
    t_big = dataHolder.t_array
    y_big = dataHolder.y_array
    print 'making large array from realizations...'
    for i in range(start + 1, start + n_realization):
        if not i%100:
            print 'relization number: ', i
        file_name = prefix + "_" + str(i) + ".pkl"
        file_address = os.path.join(folder, file_name)
        with open(file_address, 'rb') as input:
            dataHolder = pickle.load(input)
        if not dataHolder.x_array.shape[0]:
            print 'empty realization, skipping...'
            continue
        x_big = np.vstack((x_big, dataHolder.x_array))
        y_big = np.vstack((y_big, dataHolder.y_array))
        t_big = np.vstack((t_big, dataHolder.t_array))
    print 'saving big arrays'
    np.save(os.path.join(folder, 'big_x'), x_big)
    np.save(os.path.join(folder, 'big_y'), y_big)
    np.save(os.path.join(folder, 'big_t'), t_big)


def plume_location_multiple_realizations(t_target_array, folder, prefix, n_realization, attrib='x_array', start=0):
    """
    return the plume location for multiple realiztions at given times
    each row of the output is the particle cloud location at one time
    """
    #first stich together all the arrays
    #read in he first one

    file_name = prefix + '_' + str(start) + ".pkl"
    file_address = os.path.join(folder, file_name)
    with open(file_address, 'rb') as input:
        dataHolder = pickle.load(input)
    x_big = getattr(dataHolder, attrib)
    t_big = dataHolder.t_array
    print 'making large array from realizations...'
    for i in range(start+1, start+n_realization):
        file_name = prefix + "_" + str(i) + ".pkl"
        file_address = os.path.join(folder, file_name)
        with open(file_address, 'rb') as input:
            dataHolder = pickle.load(input)
        x_mat = getattr(dataHolder, attrib)
        t_mat = dataHolder.t_array
        x_big = np.vstack((x_big, x_mat))
        t_big = np.vstack((t_big, t_mat))
    print 'done'
    n_given_times = len(t_target_array)
    n_particles = t_big.shape[0]
    out_put_array = np.zeros((n_given_times, n_particles))
    for i in range(n_given_times):
        t = t_target_array[i]
        out_put_array[i, :] = plume_location_at_given_time(t, x_big, t_big)
    return out_put_array

def com_msd_at_given_time(t, x_array, time_array, com_const=0.0):
    """
    return the COM and Mean Square Difference with respect
    to the center of mass.
    """
    plumeLocation = plume_location_at_given_time(t, x_array, time_array) + com_const
    com = np.mean(plumeLocation)
    msd = np.mean(np.power(plumeLocation-com,2))
    return com, msd

def com_msd_given_location(plume_location):
    com = np.mean(plume_location)
    msd = np.mean(np.power(plume_location-com,2))
    return com, msd

def com_msd_model(t_target_array, address, attrib='x_array', com_const=0.0):
    """
    return the com and msd for a model at given times
    """
    with open(address,'rb') as input:
        dataHolder = pickle.load(input)
    x_mat = getattr(dataHolder, attrib)
    t_mat = dataHolder.t_array
    out_put_size = len(t_target_array)
    com_array = np.zeros(out_put_size)
    msd_array = np.zeros(out_put_size)
    for i in range(out_put_size):
        t = t_target_array[i]
        com_array[i], msd_array[i] = com_msd_at_given_time(t, x_mat, t_mat, com_const=com_const)
    return com_array, msd_array

def com_msd_multiple_realizations(t_target_array, folder, prefix, n_realization, attrib = 'x_array'):
    """
    return the com and msd for multiple realiztions at given times
    """
    #first stich together all the arrays
    #read in he first one
    file_name = prefix + "_0" + ".pkl"
    file_address = os.path.join(folder, file_name)
    with open(file_address, 'rb') as input:
        dataHolder = pickle.load(input)
    x_big = getattr(dataHolder, attrib)
    t_big = dataHolder.t_array
    for i in range(1,n_realization):
        file_name = prefix + "_" + str(i) + ".pkl"
        file_address = os.path.join(folder, file_name)
        with open(file_address, 'rb') as input:
            dataHolder = pickle.load(input)
        x_mat = getattr(dataHolder, attrib)
        t_mat = dataHolder.t_array
        x_big = np.vstack((x_big, x_mat))
        t_big = np.vstack((t_big, t_mat))
    out_put_size = len(t_target_array)
    com_array = np.zeros(out_put_size)
    msd_array = np.zeros(out_put_size)
    for i in range(out_put_size):
        t = t_target_array[i]
        com_array[i], msd_array[i] = com_msd_at_given_time(t, x_big, t_big)
    return com_array, msd_array

def kde_2d_multiple_times(plume_x, plume_y, X, Y):
    """
    returns the kde estimate of the 2d plume for each entry (array) in plume_x, plume_y
    :param plume_x: array containing multiple x_arrays for the plume
    :param plume_y: array containing multiple y_arrays for the plume
    :param X: x values for a grid to query the kde
    :param Y: y values for a grid to query the kde
    :return: an array of arrays containing all kde estimates
    """
    Z = []
    positions = np.vstack([X.ravel(), Y.ravel()])
    for i in range(len(plume_x)):
        values = np.vstack([plume_x[i], plume_y[i]])
        kernel = stats.gaussian_kde(values)
        Z.append(np.reshape(kernel(positions).T, X.shape))
    return Z

def choose_smaller_set(plume_x_array, plume_y_array, n_samples):
    assert(len(plume_x_array)==len(plume_y_array))
    idx_select = np.random.randint(0, plume_x_array.shape[1], n_samples)
    return plume_x_array[:,idx_select], plume_y_array[:,idx_select]


def save_plume_2d_with_kde(t_target, n_query, model, data, data_save_folder, max_samples=2000000):
    """
    This function generates and saves X,Y,Z for data and model for the same times for data and model
    :param t_target:
    :param n_query:
    :param model:
    :param data:
    :param data_save_folder:
    :param max_samples:
    :return:
    """
    plume_x_model, plume_y_model = plume_location_multiple_times(t_target, model.x_array, model.y_array, model.t_array)
    # process data
    plume_x_data, plume_y_data = plume_location_multiple_times(t_target, data.x_array, data.y_array, data.t_array)
    # using maximum 200,000 points
    if max_samples:
        if plume_x_data.shape[1] > max_samples:
            print 'using only ' + str(max_samples) + ' points from data'
            plume_x_data, plume_y_data = choose_smaller_set(plume_x_data, plume_y_data, max_samples)
        if plume_x_model.shape[1] > max_samples:
            print 'using only ' + str(max_samples) + ' points from model'
            plume_x_model, plume_y_model = choose_smaller_set(plume_x_model, plume_y_model, max_samples)

    xmax_model = np.max(plume_x_model)
    ymin_model = np.min(plume_y_model)
    ymax_model = np.max(plume_y_model)

    xmax_data = np.max(plume_x_data)
    ymin_data = np.min(plume_y_data)
    ymax_data = np.max(plume_y_data)

    xmin = 0.0
    xmax = max(xmax_model, xmax_data)
    ymin = min(ymin_model, ymin_data)
    ymax = max(ymax_model, ymax_data)

    y_center = data.y_array[0, 0]
    dy = min(abs(y_center - ymin), abs(y_center - ymax))
    X, Y = np.mgrid[xmin:xmax:n_query, ymin:ymax:n_query]
    print 'calculating kde model'
    Z2 = kde_2d_multiple_times(plume_x_model, plume_y_model, X, Y)
    print 'calculating kde data'
    Z = kde_2d_multiple_times(plume_x_data, plume_y_data, X, Y)
    save_name = 'xy_contour'
    save_path = os.path.join(data_save_folder, save_name+'.npz')
    np.savez(save_path, X=X, Y=Y)
    save_name = 'z_contour'
    save_path = os.path.join(data_save_folder, save_name + '.npz')
    np.savez(save_path, dataZ=Z, modelZ=Z2)
    save_name = 'ycorrections'
    save_path = os.path.join(data_save_folder, save_name + '.npz')
    np.savez(save_path, y_center=y_center, dy=dy)

def plot_plume_2d_from_saved(t_target, nlevels, X, Y, Z, Z2, y_center, dy, save_folder, save_name,
                           t_scale, scale_str=r'\overline{\delta t}$', l_scale=1.0, fmt='png'):
    for plot_idx in range(len(t_target)):
    # for plot_idx in [0]:
        # levels = np.linspace(0, min(np.max(Z[plot_idx]), np.max(Z2[plot_idx])), nlevels + 1)[:-1]
        levels = np.linspace(0, min(np.amax(Z[plot_idx]), np.amax(Z2[plot_idx])), nlevels+1)[:-1]
        # print levels
        fig = plt.figure(figsize=[8, 6])
        ax = fig.add_subplot(1, 1, 1)
        ax.hold(True)
        opacity = 0.7
        # cmap = plt.cm.gnuplot
        cmap = plt.cm.afmhot_r
        cmap = plt.cm.Paired_r
        cmapobj = plt.cm.get_cmap('afmhot_r')
        colors = [cmapobj(i) for i in np.linspace(0.0,1, num=nlevels+1)]
        map1 = Z[plot_idx]
        map1[map1 < 1e-6] = np.nan
        # ax.contourf(X/l_scale, Y/l_scale, map1, levels=levels, cmap=cmap, alpha=opacity)
        Y = Y - y_center
        y_center = 0.0
        ax.contourf(X/l_scale, Y/l_scale, map1, levels=levels, colors =colors, alpha=opacity)
        # ax.contour(X / l_scale, Y / l_scale, map1, levels=levels, colors = colors)
        # ax.contour(X / l_scale, Y / l_scale, map1, levels=levels, colors='b')

        map2 = Z2[plot_idx]
        map2[map2 < 1e-6] = np.nan
        ax.contour(X/l_scale, Y/l_scale, map2, levels=levels, colors='k', linestyles='dashed', lw=2.0)
        text_str = r'$t = ' + '{0:.0f}'.format(t_target[plot_idx] / t_scale) + scale_str
        ax.set_ybound([(y_center-dy)/l_scale, (y_center+dy)/l_scale])
        # ax.set_xlabel(r"$x/l$", fontsize=18)
        # ax.set_ylabel(r"$y/l$", fontsize=18)
        # ax.set_xlabel("x/l", fontsize=24)
        # ax.set_ylabel("y/l", fontsize=24)
        ax.set_xlabel("x/l")
        ax.set_ylabel("y/l")
        bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="w", lw=1)
        # text_str += '\n'+r"$colors: \; data$" + '\n' + r"$--: \;model$"
        # plt.annotate(text_str, xy=(0.75, 0.8), xycoords='axes fraction', bbox=bbox_props, fontsize=18)
        ax.annotate(text_str, xy=(1, 1), xytext=(-15, -15), fontsize=24,
                xycoords='axes fraction', textcoords='offset points',
                bbox=bbox_props, horizontalalignment='right', verticalalignment='top')
        # plt.annotate(text_str, xy=(0.77, 0.9), xycoords='axes fraction', bbox=bbox_props, fontsize=24)
        full_save_name = '2d_' + save_name + '_' + str(plot_idx) + '.'+fmt
        save_path = os.path.join(save_folder, full_save_name)
        fig.savefig(save_path, format=fmt)
        plt.close(fig)

def plot_plume_2d_with_kde(t_target, nlevels, n_query, model, data, save_folder, save_name,
                           t_scale=None, scale_str=r"\bar{\delta t}$", max_samples=2000000, l_scale=1.0, fmt='png'):
    """
    Function to make 2d plume visualizations for multiple times and save all results
    :param t_end:
    :param nsteps:
    :param nlevels:
    :param n_query:
    :param model:
    :param data:
    :param save_folder:
    :param save_name:
    :return:
    """
    # t_target = np.linspace(0, t_end, nsteps)[1:]
    # process model
    plume_x_model, plume_y_model = plume_location_multiple_times(t_target, model.x_array, model.y_array, model.t_array)
    # process data
    plume_x_data, plume_y_data = plume_location_multiple_times(t_target, data.x_array, data.y_array, data.t_array)
    #using maximum 200,000 points
    if max_samples:
        if plume_x_data.shape[1] > max_samples:
            print 'using only '+ str(max_samples) + ' points from data'
            plume_x_data, plume_y_data = choose_smaller_set(plume_x_data, plume_y_data, max_samples)
        if plume_x_model.shape[1] > max_samples:
            print 'using only ' + str(max_samples) + ' points from model'
            plume_x_model, plume_y_model = choose_smaller_set(plume_x_model, plume_y_model, max_samples)

    xmax_model = np.max(plume_x_model)
    ymin_model = np.min(plume_y_model)
    ymax_model = np.max(plume_y_model)

    xmax_data = np.max(plume_x_data)
    ymin_data = np.min(plume_y_data)
    ymax_data = np.max(plume_y_data)

    xmin = 0.0
    xmax = max(xmax_model, xmax_data)
    ymin = min(ymin_model, ymin_data)
    ymax = max(ymax_model, ymax_data)

    y_center = data.y_array[0,0]
    dy = min(abs(y_center-ymin), abs(y_center-ymax))
    X, Y = np.mgrid[xmin:xmax:n_query, ymin:ymax:n_query]
    print 'calculating kde model'
    Z2 = kde_2d_multiple_times(plume_x_model, plume_y_model, X, Y)
    print 'calculating kde data'
    Z = kde_2d_multiple_times(plume_x_data, plume_y_data, X, Y)
    for plot_idx in range(len(t_target)):
        levels = np.linspace(0, min(np.max(Z[plot_idx]), np.max(Z2[plot_idx])), nlevels + 1)[:-1]
        fig = plt.figure(figsize=[8, 6])
        ax = fig.add_subplot(1, 1, 1)
        ax.hold(True)
        ax.contourf(X/l_scale, Y/l_scale, Z[plot_idx], levels=levels, cmap=plt.cm.gnuplot)
        ax.contour(X/l_scale, Y/l_scale, Z2[plot_idx], levels=levels, colors='k', linestyles='dashed', lw=2.0)
        if t_scale:
            text_str = r"$t = " + '{0:.0f}'.format(t_target[plot_idx] / t_scale) + scale_str
        else:
            text_str = 't = ' + '{0:.0f}'.format(t_target[plot_idx])
        ax.set_ybound([(y_center-dy)/l_scale, (y_center+dy)/l_scale])
        ax.set_xlabel(r"$x/l$", fontsize=18)
        ax.set_ylabel(r"$y/l$", fontsize=18)
        bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="b", lw=2)
        text_str += '\n'+r"$colors: \; data$" + '\n' + r"$--: \;model$"
        plt.annotate(text_str, xy=(0.75, 0.8), xycoords='axes fraction', bbox=bbox_props, fontsize=18)
        full_save_name = '2d_' + save_name + '_' + str(plot_idx) + '.'+fmt
        save_path = os.path.join(save_folder, full_save_name)
        fig.savefig(save_path, format=fmt)
        plt.close(fig)


def plot_msd_com_both_one(target_time_array_model, model_com_array, model_msd_array, target_time_array_data, data_com, data_msd, save_folder,
                          save_name, save_prefix, axis_dict, data_label, model_label_array, t_scale=None, lw=1, fmt='png'):
    if t_scale:
        plot_time_data = target_time_array_data / t_scale
        plot_time_model = target_time_array_model / t_scale
    else:
        plot_time_data = target_time_array_data
        plot_time_model = target_time_array_model
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax.hold(True)
    ax2.hold(True)
    # plot the mean from the different realizations
    styles = ['-', '-o', '-s', '-v', '-^']
    styles_iter = iter(styles)
    # data
    style = next(styles_iter)
    ax.plot(plot_time_data, data_msd, style, label=data_label, lw=lw)
    ax2.plot(plot_time_data, data_com, style, label=data_label, lw=lw)

    # plot the model results
    styles = ['--', '--o', '--s', '--v', '--^']
    for i in range(len(model_com_array)):
        styles_iter = iter(styles)
        style = next(styles_iter)
        ax.plot(plot_time_model, model_msd_array[i], style, label=model_label_array[i], lw=lw)
        ax2.plot(plot_time_model, model_com_array[i], style, label=model_label_array[i], lw=lw)

    # max_msd = max(max(model_msd), max(data_msd))
    # min_msd = min(min(model_msd), min(data_msd))
    # ax.set_xbound([min_msd, max_msd])

    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    if t_scale:
        ax.set_xlabel(r"$t/ \bar{\delta t}$", fontsize=18)
        ax.set_xbound([np.amin(plot_time_data), np.amax(plot_time_data)])
    else:
        ax.set_xlabel("time")
    ax.set_ylabel(axis_dict['ylabel1'], fontsize=18)
    ax.legend(loc="best", fontsize=14)
    ax.set_yscale('log')
    ax.set_xscale('log')
    final_save_name = 'MSD_' + save_prefix + save_name + '.'+fmt
    save_path = os.path.join(save_folder, final_save_name)
    fig.savefig(save_path, format=fmt)
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    if t_scale:
        ax2.set_xlabel(r"$t/ \bar{\delta t}$", fontsize=18)
        ax2.set_xbound([np.amin(plot_time_data), np.amax(plot_time_data)])
    else:
        ax2.set_xlabel("time")
    ax2.set_ylabel(axis_dict['ylabel2'], fontsize=18)
    if 'ymin' in axis_dict:
        ax2.set_ybound([axis_dict['ymin'], axis_dict['ymax']])
    ax2.legend(loc="best", fontsize=14)
    final_save_name = 'COM_' + save_prefix + save_name + '.' + fmt
    save_path = os.path.join(save_folder, final_save_name)
    fig2.savefig(save_path, format=fmt)
    plt.close(fig)
    plt.close(fig2)

def plot_moment_inset(target_time_array_data, msd_x_data, msd_y_data, data_label, target_time_array_model,
                      msd_x_model_array, msd_y_model_array, model_label_array, t_scale, zoom_box, zoom,
                      save_folder, axis_dict, lw=1, fmt='png'):
    plot_time_data = target_time_array_data / t_scale
    plot_time_model = target_time_array_model / t_scale
    colors = ['b', 'g', 'r']
    styles = ['-', '--', '-.']
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hold(True)
    # plot the first one in the main axis
    colors_iter = iter(colors)
    styles_iter = iter(styles)
    # data
    ax.plot(plot_time_data, msd_x_data, linestyle=next(styles_iter), color=next(colors_iter),
            label=data_label, lw=lw)
    # plot the model results
    for i in range(len(msd_x_model_array)):
        ax.plot(plot_time_model, msd_x_model_array[i], linestyle=next(styles_iter), color=next(colors_iter),
                label=model_label_array[i], lw=lw)
    xbound = [10,450]
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax.set_xlabel(r"$t/ \bar{\delta t}$", fontsize=18)
    ax.set_xbound([np.amin(plot_time_data), np.amax(plot_time_data)])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xbound(xbound)
    ax.set_ylabel(axis_dict['ylabel1'], fontsize=18)
    ax.legend(loc="best", fontsize=14)
    #plot the second array in inset
    left, bottom, width, height = [0.56, 0.23, 0.37, 0.4]
    ax2 = fig.add_axes([left, bottom, width, height])
    colors_iter = iter(colors)
    styles_iter = iter(styles)
    # data
    ax2.plot(plot_time_data, msd_y_data, linestyle=next(styles_iter), color=next(colors_iter),
            label=data_label, lw=lw)
    # plot the model results
    for i in range(len(msd_x_model_array)):
        ax2.plot(plot_time_model, msd_y_model_array[i], linestyle=next(styles_iter), color=next(colors_iter),
                label=model_label_array[i], lw=lw)
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xbound(xbound)
    ax2.set_ylabel(axis_dict['ylabel2'], fontsize=14)
    ax2.set_xlabel(r"$t/ \bar{\delta t}$", fontsize=14)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    final_save_name = 'MSD_' + 'inset' + '.'+fmt
    save_path = os.path.join(save_folder, final_save_name)
    fig.savefig(save_path, format=fmt)
    plt.close(fig)

def plot_moment_zoom(target_time_array_data, msd_x_data, data_label, target_time_array_model,
                      msd_x_model_array, model_label_array, t_scale, zoom_box, zoom,
                      save_folder, save_prefix, axis_dict, lw=1, zoomloc=4, fmt='png', cor1=1, cor2=2,
                      legloc='best'):
    legend_size = 13
    plot_time_data = target_time_array_data / t_scale
    plot_time_model = target_time_array_model / t_scale
    colors = ['b', 'g', 'r']
    styles = ['-', '--', '-.']
    fig = plt.figure(figsize=[6,4])
    ax = fig.add_subplot(1, 1, 1)
    ax.hold(True)
    # plot the first one in the main axis
    colors_iter = iter(colors)
    styles_iter = iter(styles)
    # data
    ax.plot(plot_time_data, msd_x_data, linestyle=next(styles_iter), color=next(colors_iter),
            label=data_label, lw=lw)
    # plot the model results
    for i in range(len(msd_x_model_array)):
        ax.plot(plot_time_model, msd_x_model_array[i], linestyle=next(styles_iter), color=next(colors_iter),
                label=model_label_array[i], lw=lw)
    xbound = [10,450]
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    # ax.set_xlabel(r"$t/ \bar{\delta t}$", fontsize=18)
    ax.set_xlabel('nondimensional time')
    ax.set_xbound([np.amin(plot_time_data), np.amax(plot_time_data)])
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xbound(xbound)
    # ax.set_ylabel(axis_dict['ylabel1'], fontsize=18)
    ax.set_ylabel(axis_dict['ylabel1'])
    ax.legend(loc=legloc, fontsize=legend_size)
    #plot the zoomed section in inset
    left, bottom, width, height = [0.56, 0.23, 0.37, 0.4]
    ax2 = zoomed_inset_axes(ax, zoom, loc=zoomloc)
    tmin, tmax = zoom_box[0][0], zoom_box[0][1]
    pmin, pmax = zoom_box[1][0], zoom_box[1][1]
    idx_zoom_data = [i for i in range(len(plot_time_data)) if tmin < plot_time_data[i] < tmax]
    idx_zoom_model = [i for i in range(len(plot_time_model)) if tmin < plot_time_model[i] < tmax]
    colors_iter = iter(colors)
    styles_iter = iter(styles)
    # data
    ax2.plot(plot_time_data[idx_zoom_data], msd_x_data[idx_zoom_data], linestyle=next(styles_iter),
             color=next(colors_iter), label=data_label, lw=lw)
    # plot the model results
    for i in range(len(msd_x_model_array)):
        ax2.plot(plot_time_model[idx_zoom_model], msd_x_model_array[i][idx_zoom_model], linestyle=next(styles_iter),
                 color=next(colors_iter), label=model_label_array[i], lw=lw)
    ax2.set_xlim(tmin, tmax)
    ax2.set_ylim(pmin, pmax)

    # turn off the zoomed region ticks
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    # connect box
    mark_inset(ax, ax2, loc1=cor1, loc2=cor2, fc="none", ec="0.5")
    final_save_name = save_prefix + '_zoom' + '.'+fmt
    save_path = os.path.join(save_folder, final_save_name)
    fig.savefig(save_path, format=fmt)
    plt.close(fig)

def plot_plume_evolution_histogram(target_time_array, nbins,  xmin, xmax, attrib,
                                   save_folder, data_plume_list, model_plume_list, stencil_labels,
                                   data_labels, save_name, x_plot_min, x_plot_max, x_scale,
                                   t_scale=None, lw = 1, figsize=[10,4], tidx=None, fmt='png', save_pre=None,
                                   zoom=False):
    legend_size = 13
    if zoom:
        legend_size = 13
    final_plumes = []
    single_time = False
    if tidx != None:
        t_end = target_time_array[tidx]
        single_time = True
    n_given_times = len(target_time_array)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    ax.hold(True)
    bins = np.linspace(xmin,xmax,nbins)
    mid_dx = bins[:-1] + np.diff(bins) / 2
    plot_x = mid_dx/x_scale
    nn = len(target_time_array)
    if single_time:
        nn = len(stencil_labels) + 1
    # colors = cm.rainbow(np.linspace(0, 1, nn))
    master_colors = ['g', 'r', 'c', 'm', 'k', 'b']
    colors = master_colors[:nn+1]

    colors_iter = iter(colors)
    styleC = iter(["-","--",'-.',':', '-', '--'])
    artist_array = []
    label_array = []
    ## plot the plume for data sets
    for idata, plumes in enumerate(data_plume_list):
        if not single_time:
            colors_iter = iter(colors)
            time_idx = range(n_given_times)
            legend_color = 'k'
        else:
            plot_color = 'b'
            legend_color = 'b'
            # plot_color = legend_color
            time_idx = [tidx]
        style = next(styleC)
        artist = plt.Line2D((0,1),(0,0), color=legend_color, linestyle=style)
        artist_array.append(artist)
        # label_array.append(r"$"+data_labels[idata]+r"$")
        label_array.append(data_labels[idata])
        for i in time_idx:
            time = target_time_array[i]
            plume = plumes[i,:]
            n, bins = np.histogram(plume, bins=bins, density=True)
            if not single_time:
                plot_color = next(colors_iter)
            ax.plot(plot_x, n, style, color = plot_color, lw = lw,
                    label='{:.2e}'.format(time))
            final_plumes.append(n)
    ## plot the plume for models
    for imodel , plumes in enumerate(model_plume_list):
        # style = '--'
        style = next(styleC)
        if not single_time:
            colors_iter = iter(colors)
            # style = next(styleC)
            legend_color = 'k'
            time_idx = range(n_given_times)
        else:
            legend_color = next(colors_iter)
            plot_color = legend_color
            time_idx = [tidx]
        artist = plt.Line2D((0,1),(0,0), color=legend_color, linestyle=style)
        artist_array.append(artist)
        # label_array.append(r"$"+stencil_labels[imodel]+r"$")
        label_array.append(stencil_labels[imodel])
        for i in time_idx:
            time = target_time_array[i]
            plume = plumes[i,:]
            n, bins = np.histogram(plume, bins=bins, density=True)
            if not single_time:
                plot_color = next(colors_iter)
            ax.plot(plot_x, n, style, color = plot_color, lw=lw,
                    label='{:.2e}'.format(time))
            final_plumes.append(n)
    if attrib=='x':
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
        # ax.set_xlabel(r"$x/l$", fontsize=18)
        ax.set_xlabel('x/l')
        save_txt = 'x_'
    else:
        # ax.set_xlabel(r"$y/l$", fontsize=18)
        ax.set_xlabel('y/l')
        save_txt = 'y_'
    if save_pre is not None:
        save_txt = save_pre + '_' + save_txt
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    # ax.set_ylabel(r"$pdf(C)$", fontsize=18)
    ax.set_ylabel('particle density')
    if single_time:
        l1 = 'best'
    elif  (save_pre is not None):
        l1 = 2
    else:
        l1 = 1
    if single_time and t_scale:
        # legend_title = r"$t/ \bar{\delta t}=" + '{0:.0f}'.format(t_end / t_scale) + r"$"
        legend_title = r"$t/ \overline{\delta t}=" + '{0:.0f}'.format(t_end / t_scale) + r"$"
        legend1 = plt.legend(artist_array, label_array, loc=l1, fontsize = legend_size, title=legend_title)
        plt.setp(legend1.get_title(), fontsize=legend_size)
    else:
        legend1 = plt.legend(artist_array, label_array, loc=l1, fontsize = legend_size)
    # if save_pre is None:
    plt.gca().add_artist(legend1)

    x_plot_max = min(1.5*max(plume), x_plot_max)
    ax.set_xbound([x_plot_min, x_plot_max])
    #create time legend
    if not single_time:
        artist_array2 = []
        label_array2 = []
        colors_iter = iter(colors)
        for i in range(nn):
            artist = plt.Line2D((0,1),(0,0), color=next(colors_iter), linestyle='-')
            artist_array2.append(artist)
            time = target_time_array[i]
            if t_scale:
                # label_str = r"$t/ \bar{\delta t} = " + '{0:.0f}'.format(time/t_scale)+ r"$"
                label_str = r"$t/ \overline{\delta t} = " + '{0:.0f}'.format(time / t_scale) + r"$"
            else:
                label_str = "t = "+'{:.2e}'.format(time)
            label_array2.append(label_str)
        loc2 = 4
        if save_pre is not None:
            loc2 = 1
        legend2 = plt.legend(artist_array2, label_array2, loc=loc2, fontsize = legend_size)
        plt.gca().add_artist(legend2)
    if single_time:
        if t_scale:
            last_str = str(t_end/t_scale).split('.')[0]
        else:
            last_str = str(t_end).split('.')[0]
        save_name = 'spread_' + save_txt + save_name + '_oneTime_' + last_str + '.' + fmt
    else:
        save_name = 'spread_' + save_txt + save_name + '.' + fmt
    if zoom and attrib=='x':
        axins = zoomed_inset_axes(ax, 5.0, bbox_to_anchor=[20,4.8e-3], bbox_transform=ax.transData, loc =6)  # zoom = 6
        x_zoom = 15.0
        #plot the zoomed section of data and models
        idx_zoom = plot_x < x_zoom
        x_array_zoom = plot_x[idx_zoom]
        colors = iter(['b', 'g', 'r'])
        styleC = iter(["-", "--", '-.'])
        for i in range(3):
            style = next(styleC)
            plot_color = next(colors)
            hist_zoom = final_plumes[i][idx_zoom]
            axins.plot(x_array_zoom, hist_zoom, style, color = plot_color)
        # sub region of the original image
        x1, x2, y1, y2 = 0.0, x_zoom, 0.0, 1.5*0.001
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        # turn off the zoomed region ticks
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        #connect box
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    save_path = os.path.join(save_folder, save_name)
    fig.savefig(save_path,format=fmt)
    plt.close(fig)


def get_dt_stats(input_folder, n_realz, prefix='real', verbose=True):
    dt_array = np.array([])
    for j in range(n_realz):
        if verbose:
            print "reading realization nr ", j
        case_name = prefix + "_" + str(j) + ".pkl"
        input_file = os.path.join(input_folder, case_name)
        with open(input_file, 'rb') as input:
            dataHolder = pickle.load(input)
        dt_matrix = np.diff(dataHolder.t_array)
        last_idx = dataHolder.last_idx_array
        tmp = [dt_matrix[i, :last_idx[i]] for i in range(len(last_idx))]
        tmp2 = [item for sublist in tmp for item in sublist]
        dt_array = np.hstack((dt_array, tmp2))
    return dt_array

def check_dt_stats(input_folder, save_folder, save_name, n_realz, prefix='real'):
    dt_array = get_dt_stats(input_folder, n_realz, prefix)
    fig, ax = plt.subplots(1,1)
    dt_med = np.median(dt_array)
    dt_mean = np.mean(dt_array)
    dt_p99= np.percentile(dt_array, 99)
    gkde = stats.kde.gaussian_kde(dt_array)
    query_points = np.linspace(0, 10*dt_mean, 100)
    pdf_query = gkde.evaluate(query_points)
    ax.plot(query_points, pdf_query, lw=1.5)
    ax.ticklabel_format(axis='x', style = 'sci', scilimits=(-2,2))
    ax.set_xlabel('dt')
    ax.set_ylabel('frequency')
    text_str = 'mean = ' + '{0:.1e}'.format(dt_mean)
    text_str += '\n'
    text_str += 'median = ' + '{0:.1e}'.format(dt_med) + '\n'
    text_str += 'p99 = ' + '{0:.1e}'.format(dt_p99)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="white", ec="b", lw=1)
    plt.annotate(text_str, xy=(0.5, 0.7), xycoords='axes fraction', bbox=bbox_props, size = 14)
    save_path = os.path.join(save_folder, 'dt_distrib_'+save_name+'.png')
    fig.savefig(save_path, format='png')

def compare_trans_mat_vtheta(trans_matrix_1, trans_matrix_2, figure_save_folder, prefix, refsize = 16, fmt='pdf'):
    import matplotlib.colors as colors
    fontsize = refsize * 0.8
    # plt.rcParams.update({'font.size':fontsize})
    # plt.rc('xtick', labelsize=fontsize * 0.8)
    # plt.rc('ytick', labelsize=fontsize * 0.8)
    fig = plt.figure(figsize=(9,4))
    # gs = gridspec.GridSpec(1, 3, width_ratios=[10, 10, 1])
    # ax = plt.subplot(gs[0])
    # ax2 = plt.subplot(gs[1])
    # ax_cmap = plt.subplot(gs[2])

    gs = gridspec.GridSpec(1, 2)
    gs.update(left=0.1, right=0.86, hspace=0.1)
    gs1 = gridspec.GridSpec(1, 1)
    gs1.update(left=0.89, right=0.92)
    ax = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax_cmap = plt.subplot(gs1[0])
    # ax = fig.add_subplot(1,2,1)
    # ax2 = fig.add_subplot(1,2,2)
    cmap = plt.cm.afmhot_r
    max_prob = 1.0
    # print max_prob
    p = ax.pcolor(np.sqrt(trans_matrix_1), norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=0, vmax=max_prob),
                  cmap=cmap, linewidth=0, rasterized=True)
    p2 = ax2.pcolor(np.sqrt(trans_matrix_2), norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=0, vmax=max_prob),
                  cmap=cmap, linewidth=0, rasterized=True)
    # p = ax.pcolor(trans_matrix_1, norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03), cmap=cmap, linewidth=0,
    #                                                      rasterized=True)
    # p2 = ax2.pcolor(trans_matrix_2, norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03), cmap=cmap, linewidth=0,
    #                                                        rasterized=True)
    cbar = fig.colorbar(p, ax_cmap, orientation='vertical')
    tick_array = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    cbar.set_ticks(tick_array)
    # cbar.set_ticklabels()
    lp  = 1
    ax.set_xlabel('previous class ' + r'($v_n$)', labelpad=lp)
    ax.set_ylabel('next class '+ r'($v_{n+1}$)', labelpad=lp)
    ax2.set_xlabel('previous class '+ r'($\theta_n$)', labelpad=lp)
    ax2.set_ylabel('next class '+ r'($\theta_{n+1}$)', labelpad=lp)
    mat_size = trans_matrix_1.shape[1]
    ax_bound = [0, mat_size]
    ax.set_xbound(ax_bound)
    ax.set_ybound(ax_bound)
    mat_size = trans_matrix_2.shape[1]
    ax_bound = [0, mat_size]
    ax2.set_xbound(ax_bound)
    ax2.set_ybound(ax_bound)
    fig_name = prefix +'.'+fmt
    file_name = os.path.join(figure_save_folder, fig_name)
    fig.savefig(file_name, format=fmt)
    plt.close(fig)

def compare_trans_mat(trans_matrix_1, trans_matrix_2, lag, figure_save_folder, prefix, refsize = 16, fmt='pdf'):
    import matplotlib.colors as colors
    plt.rcParams.update({'figure.autolayout': True})
    fontsize = refsize * 0.8
    # plt.rcParams.update({'font.size':fontsize})
    # plt.rc('xtick', labelsize=fontsize * 0.8)
    # plt.rc('ytick', labelsize=fontsize * 0.8)
    trans_matrix_markov_2 = copy(trans_matrix_1)
    for i in range(lag-1):
        trans_matrix_markov_2 = np.dot(trans_matrix_1,trans_matrix_markov_2)
    fig = plt.figure(figsize=(9,4.0))
    # ax = fig.add_subplot(1,2,1)
    # ax2 = fig.add_subplot(1,2,2)

    # gs = gridspec.GridSpec(1, 3, width_ratios=[10, 10, 1])
    # ax = plt.subplot(gs[0])
    # ax2 = plt.subplot(gs[1])
    # ax_cmap = plt.subplot(gs[2])

    gs = gridspec.GridSpec(1, 2)
    gs.update(left= 0.1, right=0.86, hspace = 0.1)
    gs1 = gridspec.GridSpec(1,1)
    gs1.update(left=0.89, right = 0.92)
    ax = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax_cmap = plt.subplot(gs1[0])
    cmap = plt.cm.afmhot_r
    max_prob = np.amax(trans_matrix_1)
    max_prob = 1.0
    p = ax.pcolor(np.sqrt(trans_matrix_2), norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=0, vmax=max_prob),
                  cmap=cmap, linewidth=0, rasterized=True)
    p2 = ax2.pcolor(np.sqrt(trans_matrix_markov_2), norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,vmin=0, vmax=max_prob),
                  cmap=cmap, linewidth=0, rasterized=True)
    # p = ax.pcolor(trans_matrix_2, norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=0, vmax=max_prob),
    #               cmap=cmap, linewidth=0, rasterized=True)
    # p2 = ax2.pcolor(trans_matrix_markov_2,
    #                 norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03, vmin=0, vmax=max_prob),
    #                 cmap=cmap, linewidth=0, rasterized=True)
    cbar = fig.colorbar(p, ax_cmap, orientation='vertical')
    tick_array = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    cbar.set_ticks(tick_array)
    # plt.tight_layout()
    # fig.colorbar(p, cax=ax_cmap)
    if prefix.startswith('v'):
        prev = r'($v_n$)'
        next = r'($v_{n+1}$)'
        next_str = 'next velocity class'
    else:
        prev = r'($\theta_n$)'
        next = r'($\theta_{n+1}$)'
        next_str = 'next angle (radians)'
    lp  = 1
    ax.set_xlabel('previous class '+prev, labelpad=lp)
    ax.set_ylabel('next class '+next, labelpad=lp)
    ax2.set_xlabel('previous class '+prev, labelpad=lp)
    ax2.set_ylabel('next class '+next, labelpad=lp)
    mat_size = trans_matrix_1.shape[1]
    ax_bound = [0, mat_size]
    ax.set_xbound(ax_bound)
    ax.set_ybound(ax_bound)
    ax2.set_xbound(ax_bound)
    ax2.set_ybound(ax_bound)
    if prefix == 'theta':
        ax_bound = [-np.pi, np.pi]
    fig_name = prefix + '_' + 'matrix_compare' +'.'+fmt
    file_name = os.path.join(figure_save_folder, fig_name)
    fig.savefig(file_name, format=fmt)
    plt.close(fig)
    col_array = np.floor(np.linspace(1,mat_size-2,6))
    fontsize = refsize
    # plt.rcParams.update({'font.size': fontsize})
    # plt.rc('xtick', labelsize=fontsize)
    # plt.rc('ytick', labelsize=fontsize)
    # for col in col_array:
    #     col = int(col)
    #     fig2 = plt.figure(figsize=[6,4])
    #     ax = fig2.add_subplot(1,1,1)
    #     if prefix == 'theta':
    #         label_str = r"$T_5^{\theta}(i,j)$"
    #         xvals = np.linspace(-np.pi, np.pi, num=trans_matrix_2.shape[0])
    #         assert(len(xvals)==trans_matrix_2.shape[0])
    #         ax.plot(xvals, trans_matrix_2[:, col], label=label_str)
    #     else:
    #         label_str = r"$T_5^{v}(i,j)$"
    #         ax.plot(trans_matrix_2[:, col], label=label_str)
    #     ax.hold(True)
    #     if prefix == 'theta':
    #         label_str = r"${T_1^{\theta}(i,j)}^5$"
    #         ax.plot(xvals, trans_matrix_markov_2[:, col], label=label_str)
    #     else:
    #         label_str = r"${T_1^{v}(i,j)}^5$"
    #         ax.plot(trans_matrix_markov_2[:, col], label=label_str)
    #     ax.axvline(col, color='r', linestyle='--')
    #     # ax.set_title(r"$j = " + '{0:.0f}'.format(col) +r" $")
    #     ax.legend(fontsize=13, loc='best')
    #     ax.set_xlabel(next_str)
    #     ax.set_ylabel("probability")
    #     ax.set_xbound(ax_bound)
    #     ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    #     fig_name = prefix + '_' + str(col).split('.')[0] + '.' + fmt
    #     file_name = os.path.join(figure_save_folder, fig_name)
    #     fig2.savefig(file_name, format=fmt)
    #     plt.close(fig2)

def compare_trans_mat_hist(trans_matrix_1, trans_matrix_2, lag, figure_save_folder, prefix, refsize=16,
                           legend_size=13, fmt='pdf', col_array=None):
    # plt.rcParams.update({'font.size': refsize})
    # plt.rc('xtick', labelsize=refsize)
    # plt.rc('ytick', labelsize=refsize)
    trans_matrix_markov_2 = copy(trans_matrix_1)
    for i in range(lag-1):
        trans_matrix_markov_2 = np.dot(trans_matrix_1,trans_matrix_markov_2)
    if prefix.startswith('v'):
        next_str = 'next velocity class'
        label_str1 = r"$T_5^{v}(i,j)$"
        label_str2 = r"${T_1^{v}(i,j)}^5$"
    else:
        next_str = 'next angle class (radians)'
        label_str1 = r"$T_5^{\theta}(i,j)$"
        label_str2 = r"${T_1^{\theta}(i,j)}^5$"
    mat_size = trans_matrix_1.shape[0]
    if not col_array:
        col_array = np.floor(np.linspace(1,mat_size-2,6))
        col_array = np.hstack(([0,1,2,3,4,5,6], col_array))
    if prefix == 'theta':
        ax_bound = [-np.pi, np.pi]
        index = np.linspace(-np.pi, np.pi, num=mat_size)
    else:
        ax_bound = [0,mat_size]
        index = np.linspace(0,mat_size,num=mat_size)
    bin_width = (index[1]-index[0])
    opacity = 0.5
    for col in col_array:
        col = int(col)
        fig2 = plt.figure(figsize=[6,4])
        ax = fig2.add_subplot(1,1,1)
        label_str = r"$data,\; lag=" + '{0:.0f}'.format(lag)+ r" $"
        # ax.bar(index, trans_matrix_2[:,col], bin_width, color='b', alpha=opacity, label = label_str)
        ax.step(index, trans_matrix_2[:,col], where='mid', label= label_str1)
        ax.hold(True)
        label_str = r"$T(i,j)^" + '{0:.0f}'.format(lag)+ r" $"
        # ax.bar(index, trans_matrix_markov_2[:,col], bin_width, color = 'g', alpha=opacity, label = label_str)
        ax.step(index, trans_matrix_markov_2[:, col], 'g--',where='mid', label=label_str2)
        ax.legend(fontsize=legend_size, loc='best')
        ax.set_xlabel(next_str)
        ax.set_ylabel("probability")
        ax.set_xbound(ax_bound)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        fig_name = prefix + '_' + str(col).split('.')[0] +'_hist'+ '.' + fmt
        file_name = os.path.join(figure_save_folder, fig_name)
        fig2.savefig(file_name, format=fmt)
        plt.close(fig2)

def plot_bt(input_array, label_array, t_scale, l_frac, figure_save_folder, fmt='png', lw=1.0, zoom=False,
            legend_size=13, bound_box=None):
    final_cdf = []
    style_iter = iter(['-', '--', '-.', ':', '--', '-.'])
    marker_iter = iter([None, None, None, None, None, None])
    color_iter = iter(['b', 'g', 'r', 'c', 'm', 'k'])

    artist_array = []
    fig = plt.figure(figsize=[6,4])
    ax = fig.add_subplot(1,1,1)
    big_t = np.array([])
    #construct big_t used for making bins for the bt curve
    for t_array in input_array:
        big_t = np.hstack((big_t, t_array / t_scale))
    # make bins
    t_start = np.amin(big_t)
    t_end = np.percentile(big_t, 99.8)
    # t_end = min(t_end, 3e3)
    # t_end = np.amax(big_t)
    t_edges = np.linspace(t_start, t_end, 150)
    for idx, t_array in enumerate(input_array):
        # all times should be greater than zero
        t_array = t_array[t_array>0]
        t_center_vals, t_cdf = get_cdf_from_bins(t_array / t_scale, t_edges)
        final_cdf.append(t_cdf)
        color = next(color_iter)
        style = next(style_iter)
        marker = next(marker_iter)
        artist = plt.Line2D((0, 1), (0, 0), color=color, linestyle=style, marker=marker)
        artist_array.append(artist)
        ax.plot(t_center_vals, t_cdf, style, color=color, lw=lw, marker=marker)
    legend_title = r"$x_{t} = " + '{0:.2f}'.format(l_frac) + "L$"
    legend1 = plt.legend(artist_array, label_array, loc=4, fontsize=legend_size, title=legend_title)
    plt.setp(legend1.get_title(), fontsize=legend_size)
    # ax.set_xlabel(r"$t_b/ \overline{\delta t}$", fontsize=18)
    # ax.set_ylabel(r"$CDF$", fontsize=18)
    ax.set_xlabel('nondimensional FPT')
    ax.set_ylabel('cumulative distribution')

    # legend1 = plt.legend(artist_array, label_array, loc=4, fontsize=24, title=legend_title)
    # ax.set_xlabel(r"$t_b/ \overline{\delta t}$", fontsize=28)
    # ax.set_ylabel(r"$CDF$", fontsize=18)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    plt.gca().add_artist(legend1)
    # plt.setp(legend1.get_title(), fontsize=24)
    frac_str = str(l_frac).split('.')[1]
    ax.set_ybound([0,1.02])
    if bound_box:
        ax.set_xbound(bound_box)
    zoomstr = ''
    if zoom:
        zoomstr = '_zoom'
        tmin, tmax = 550, 750
        pmin, pmax = 0.9, 1.0
        axins = zoomed_inset_axes(ax, 4.0, bbox_to_anchor=[1100,0.45], bbox_transform=ax.transData, loc =8)  # zoom = 6
        # axins = zoomed_inset_axes(ax, 4.5, loc=10)  # zoom = 6
        #plot the zoomed section of data and models
        idx_zoom = [i for i in range(len(t_center_vals)) if tmin<t_center_vals[i]<tmax]
        x_array_zoom = t_center_vals[idx_zoom]
        colors = iter(['b', 'g', 'r'])
        styleC = iter(["-", "--", '-.'])
        for i in range(3):
            style = next(styleC)
            plot_color = next(colors)
            hist_zoom = final_cdf[i][idx_zoom]
            axins.plot(x_array_zoom, hist_zoom, style, color = plot_color)
        # sub region of the original image
        x1, x2, y1, y2 = tmin, tmax, pmin, pmax
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        # turn off the zoomed region ticks
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        #connect box
        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")
    if frac_str == '0':
        fract_str = '1'
    fig_name = 'bt_linear_' + frac_str + zoomstr + '.' + fmt
    file_name = os.path.join(figure_save_folder, fig_name)
    fig.savefig(file_name, format=fmt)
    plt.close(fig)

def plot_bt_logscale(input_array, label_array, t_scale, l_frac, figure_save_folder, fmt='png', lw=1):
    style_iter = iter(['-', '--', '-.'])
    color_iter = iter(['b', 'g', 'r'])
    artist_array = []
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    big_t = np.array([])
    for t_array in input_array:
        big_t = np.hstack((big_t, t_array/t_scale))
    #remove zero times...
    big_t = big_t[big_t != 0]
    # make bins
    t_log_edges = abs_vel_log_bins_low_high(big_t, 100, max_allowed=0.5)
    for idx, t_array in enumerate(input_array):
        #remove t = 0
        t_array = t_array[t_array>0]
        t_center_vals, t_cdf = get_cdf_from_bins(np.log(t_array / t_scale), t_log_edges)
        color = next(color_iter)
        style = next(style_iter)
        artist = plt.Line2D((0, 1), (0, 0), color=color, linestyle=style)
        artist_array.append(artist)
        ax.plot(np.exp(t_center_vals), t_cdf, style, color=color, lw=lw)
    legend_title = r"$x_{t} = " + '{0:.2f}'.format(l_frac) + "L$"
    legend1 = plt.legend(artist_array, label_array, loc=4, fontsize=13, title=legend_title)
    plt.gca().add_artist(legend1)
    ax.set_xlabel(r"$t_b/ \overline{\delta t}$", fontsize=18)
    ax.set_ylabel(r"$CDF$", fontsize=18)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax.set_xscale('log')
    ax.set_ybound([0, 1.02])
    ax.set_xbound([np.amin(big_t), np.amax(big_t)])
    plt.gca().add_artist(legend1)
    frac_str = str(l_frac).split('.')[1]
    if frac_str == '0':
        fract_str = '1'
    fig_name = 'bt_logscale_' + frac_str + '.' + fmt
    file_name = os.path.join(figure_save_folder, fig_name)
    fig.savefig(file_name, format=fmt)
    plt.close(fig)


def plot_plume_x_side_y_oneTime(t_target, nxbins, nybins, xmin, xmax, ymin, ymax,
                        data_x_plume, data_y_plume, model_x_plume_list, model_y_plume_list, model_label_list,
                        t_scale, l_scale, lw, fmt, save_folder):
    legend_size = 13
    # style_iter = iter(['-', '--', '-.'])
    # color_iter = iter(['b', 'g', 'r'])

    color_iter = iter(['b', 'g', 'r', 'c', 'm', 'k'])
    style_iter = iter(["-", "--", '-.', ':', '--', '-.'])

    fig = plt.figure(figsize=[12,4])
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1,2,2)
    xbins = np.linspace(xmin, xmax, nxbins)
    mid_dx = xbins[:-1] + np.diff(xbins) / 2
    plot_x = mid_dx / l_scale

    ybins = np.linspace(ymin, ymax, nybins)
    mid_dy = ybins[:-1] + np.diff(ybins) / 2
    plot_y = mid_dy / l_scale

    label_array = []
    artist_array = []
    #plot the data
    style = next(style_iter)
    plot_color = next(color_iter)
    artist = plt.Line2D((0,1),(0,0), color=plot_color, linestyle=style)
    artist_array.append(artist)
    label_array.append(r"$data$")
    c_data_x, bins = np.histogram(data_x_plume, bins=xbins, density=True)
    ax.plot(plot_x, c_data_x, style, color = plot_color, lw = lw)
    c_data_y, bins = np.histogram(data_y_plume, bins=ybins, density=True)
    ax2.plot(plot_y, c_data_y, style, color = plot_color, lw = lw)

    for model_plume_x, model_plume_y, model_label in zip(model_x_plume_list, model_y_plume_list, model_label_list):
        style = next(style_iter)
        plot_color = next(color_iter)
        artist = plt.Line2D((0, 1), (0, 0), color=plot_color, linestyle=style)
        artist_array.append(artist)
        label_array.append(model_label)
        c_model_x, bins = np.histogram(model_plume_x, bins=xbins, density=True)
        ax.plot(plot_x, c_model_x, style, color=plot_color, lw=lw)
        c_model_y, bins = np.histogram(model_plume_y, bins=ybins, density=True)
        ax2.plot(plot_y, c_model_y, style, color=plot_color, lw=lw)

    #plot the lengend on ax2
    # legend_title = r"$t/ \bar{\delta t}=" + '{0:.0f}'.format(t_target / t_scale) + r"$"
    legend_title = r"$t/ \overline{\delta t}=" + '{0:.0f}'.format(t_target / t_scale) + r"$"
    legend1 = plt.legend(artist_array, label_array, loc=2, fontsize=legend_size, title=legend_title)
    plt.gca().add_artist(legend1)
    plt.setp(legend1.get_title(), fontsize=legend_size)
    ax.set_xbound([0.0, xmax/l_scale])
    ax2.set_xbound([ymin/l_scale, ymax/l_scale])
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    # ax.set_xlabel(r"$x/l$", fontsize=18)
    # ax2.set_xlabel(r"$y/l$", fontsize=18)
    # ax.set_ylabel(r"$pdf(C)$", fontsize=18)
    ax.set_xlabel('x/l')
    ax2.set_xlabel('y/l')
    ax.set_ylabel('particle density')
    t_str = str(t_target/t_scale).split('.')[0]
    save_name = 'xy_side_onTime_' + t_str + '.' + fmt
    save_path = os.path.join(save_folder, save_name)
    fig.savefig(save_path, format=fmt)
    plt.close(fig)


def plot_plume_x_side_y(t_target_array, nxbins, nybins, xmin, xmax, ymin, ymax,
                        data_x_plume, data_y_plume, model_x_plume_list, model_y_plume_list, model_label_list,
                        t_scale, l_scale, lw, fmt, save_folder):
    legend_size = 13
    # color_list = ['g', 'r', 'c', 'b', 'm', 'k']
    # style_list = ['--', '-.', ':']
    color_list = ['g', 'r', 'c', 'b', 'm', 'k']
    style_list = ["--", '-.', ':', '--', '-.', ':']
    color_iter = iter(color_list)
    fig = plt.figure(figsize=[12,4])
    ax = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1,2,2)
    xbins = np.linspace(xmin, xmax, nxbins)
    mid_dx = xbins[:-1] + np.diff(xbins) / 2
    plot_x = mid_dx / l_scale

    ybins = np.linspace(ymin, ymax, nybins)
    mid_dy = ybins[:-1] + np.diff(ybins) / 2
    plot_y = mid_dy / l_scale
    label_time_list = []
    artist_time_list = []
    ax.hold(True)
    ax2.hold(True)
    for i in range(len(t_target_array)):
        t_target = t_target_array[i]
        # t_str = r'$t/ \overline{\delta t}= ' +  '{0:.0f}'.format(t_target/t_scale) + "$"
        t_str = r'$t/ \overline{\delta t}= ' + '{0:.0f}'.format(t_target / t_scale) + "$"
        label_time_list.append(t_str)
        label_array = []
        artist_array = []
        #plot the data
        style = '-'
        style_iter = iter(style_list)
        plot_color = next(color_iter)
        artist = plt.Line2D((0,1),(0,0), color='k', linestyle=style)
        artist_array.append(artist)
        artist_time = plt.Line2D((0,1),(0,0), color=plot_color, linestyle='-')
        artist_time_list.append(artist_time)
        label_array.append(r"$data$")
        c_data_x, bins = np.histogram(data_x_plume[0][i,:], bins=xbins, density=True)
        ax.plot(plot_x, c_data_x, style, color = plot_color, lw = lw)
        c_data_y, bins = np.histogram(data_y_plume[0][i,:], bins=ybins, density=True)
        ax2.plot(plot_y, c_data_y, style, color = plot_color, lw = lw)
        for j in range(len(model_x_plume_list)):
            style = next(style_iter)
            model_plume_x = model_x_plume_list[j][i,:]
            model_plume_y = model_y_plume_list[j][i, :]
            model_label = model_label_list[j]
            artist = plt.Line2D((0, 1), (0, 0), color='k', linestyle=style)
            artist_array.append(artist)
            label_array.append(model_label)
            c_model_x, bins = np.histogram(model_plume_x, bins=xbins, density=True)
            ax.plot(plot_x, c_model_x, style, color=plot_color, lw=lw)
            c_model_y, bins = np.histogram(model_plume_y, bins=ybins, density=True)
            ax2.plot(plot_y, c_model_y, style, color=plot_color, lw=lw)

    ax.set_xbound([0.0, xmax / l_scale])
    ax2.set_xbound([ymin / l_scale, ymax / l_scale])
    #plot the lengend on ax2
    legend1 = plt.legend(artist_array, label_array, loc=1, fontsize=legend_size)
    plt.gca().add_artist(legend1)
    legend2 = plt.legend(artist_time_list, label_time_list, loc=2, fontsize=legend_size)
    plt.gca().add_artist(legend2)

    ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax2.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    # ax.set_xlabel(r"$x/l$", fontsize=18)
    # ax2.set_xlabel(r"$y/l$", fontsize=18)
    # ax.set_ylabel(r"$pdf(C)$", fontsize=18)
    ax.set_xlabel('x/l')
    ax2.set_xlabel('y/l')
    ax.set_ylabel('particle density')
    t_str = str(t_target/t_scale).split('.')[0]
    save_name = 'xy_side' + '.' + fmt
    save_path = os.path.join(save_folder, save_name)
    fig.savefig(save_path, format=fmt)
    plt.close(fig)