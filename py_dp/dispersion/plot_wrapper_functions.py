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
import cPickle as cPickle
import pickle as pickle
import os
import matplotlib.pyplot as plt
from py_dp.dispersion.dispersion_visualization_tools import plume_msd_com_multiple_times, \
    plume_location_multiple_times, plume_bt_multiple_locations, save_plume_2d_with_kde, plot_moment_zoom, \
    plot_plume_evolution_histogram, plot_plume_x_side_y, plot_plume_x_side_y_oneTime, plot_bt, \
    plot_plume_2d_from_saved, plot_msd_com_both_one, plot_plume_2d_with_kde

def generate_plot_data(t_end, t_scale, stencil_dt, data_save_folder, model_array, data, l, theta,
                       n_points_moments=12, n_steps_plumes=5, moments=True, plumes=True, bt=True, two_d=False,
                       kdepoints = 200000, n_pores=500, bt_bound_box=None):
    if not stencil_dt:
        stencil_dt = 1e-3
    # save t_end, t_scale, stencil_dt. Useful for the plots.
    time_file_path = os.path.join(data_save_folder, 'time_file.npz')
    np.savez(time_file_path, t_end=t_end, t_scale=t_scale, stencil_dt=stencil_dt)
    # save l. theta
    network_spec_file = os.path.join(data_save_folder, 'network_specs.npz')
    np.savez(network_spec_file, l=l, theta=theta)
    xmax = n_pores * l * np.cos(theta)
    dt_mean = t_scale
    target_time_array = np.linspace(stencil_dt, t_end, n_steps_plumes)[1:]
    target_time_array = np.floor(target_time_array / (10.0 * dt_mean)) * 10.0 * dt_mean
    # plot the evolution first and second moment of the plume
    if moments:
        n_points = n_points_moments
        print 'calculating plume moments for ' + str(n_points) + ' times'
        target_time_array_data = np.linspace(0.0, np.log(t_end / dt_mean), n_points)
        target_time_array_data = np.hstack((0.0, np.exp(target_time_array_data))) * dt_mean

        target_time_array_model = np.linspace(np.log(stencil_dt / dt_mean), np.log(t_end / dt_mean), n_points)
        target_time_array_model = np.exp(target_time_array_model) * dt_mean

        com_x_model_array = []
        com_y_model_array = []
        msd_x_model_array = []
        msd_y_model_array = []
        for model in model_array:
            com_x_model, msd_x_model, com_y_model, msd_y_model = plume_msd_com_multiple_times(target_time_array_model,
                                                                                              model)
            com_x_model_array.append(com_x_model)
            com_y_model_array.append(com_y_model)
            msd_x_model_array.append(msd_x_model)
            msd_y_model_array.append(msd_y_model)
        com_x_data, msd_x_data, com_y_data, msd_y_data = plume_msd_com_multiple_times(target_time_array_data, data)
        save_name = 'model_moments'
        save_path = os.path.join(data_save_folder, save_name+'.npz')
        np.savez(save_path, com_x=com_x_model_array, com_y=com_y_model_array, msd_x=msd_x_model_array,
                 msd_y=msd_y_model_array)
        save_name = 'data_moments'
        save_path = os.path.join(data_save_folder, save_name + '.npz')
        np.savez(save_path, com_x=com_x_data, com_y=com_y_data, msd_x=msd_x_data, msd_y=msd_y_data)
    if plumes:
        print "calculating the plume spreading in x and y direction"
        n_steps = n_steps_plumes
        data_plume_x_array = []
        data_plume_y_array = []
        data_labels = []
        #no loop needed for data
        xtemp, ytemp = plume_location_multiple_times(target_time_array, data.x_array,
                                                     data.y_array, data.t_array)
        data_plume_x_array.append(xtemp)
        data_plume_y_array.append(ytemp)
        data_labels.append(data.label)
        #loop for model
        model_plume_x_array = []
        model_plume_y_array = []
        stencil_labels = []
        for model in model_array:
            xtemp, ytemp = plume_location_multiple_times(target_time_array, model.x_array,
                                                         model.y_array, model.t_array)
            model_plume_x_array.append(xtemp)
            model_plume_y_array.append(ytemp)
            stencil_labels.append(model.label)
        save_name = 'data_plumes'
        save_path = os.path.join(data_save_folder, save_name + '.pkl')
        out = [data_plume_x_array, data_plume_y_array]
        with open(save_path, 'wb') as outfile:
            cPickle.dump(out, outfile, cPickle.HIGHEST_PROTOCOL)
        save_name = 'model_plumes'
        save_path = os.path.join(data_save_folder, save_name+'.pkl')
        out = [model_plume_x_array, model_plume_y_array]
        with open(save_path, 'wb') as outfile:
            cPickle.dump(out, outfile, cPickle.HIGHEST_PROTOCOL)
    if bt:
        print 'calculating time to get to a location'
        l_frac_array  = np.array([0.25, 0.5, 0.75])
        target_x_array = l_frac_array*xmax
        data_bt_array = []
        data_labels = []
        #no loop needed for data
        # for data in data_array:
        ttemp = plume_bt_multiple_locations(target_x_array, data.x_array, data.t_array)
        data_bt_array.append(ttemp)
        data_labels.append(data.label)
        model_bt_array = []
        stencil_labels = []
        for model in model_array:
            ttemp = plume_bt_multiple_locations(target_x_array, model.x_array,
                                                model.t_array)
            model_bt_array.append(ttemp)
            stencil_labels.append(model.label)
        save_name = 'data_bt'
        save_path = os.path.join(data_save_folder, save_name + '.pkl')
        with open(save_path, 'wb') as outfile:
            cPickle.dump(data_bt_array, outfile, cPickle.HIGHEST_PROTOCOL)
        save_name = 'model_bt'
        save_path = os.path.join(data_save_folder, save_name + '.pkl')
        with open(save_path, 'wb') as outfile:
            cPickle.dump(model_bt_array, outfile, cPickle.HIGHEST_PROTOCOL)
    if two_d:
        #plot the average plume in 2d
        print 'generating 2d plume data only for the first model...'
        plt.rc('text', usetex=False)
        n_query = 100j
        model = model_array[0]
        save_plume_2d_with_kde(target_time_array, n_query, model, data, data_save_folder, max_samples=kdepoints)

def plot_wrapper_with_saved_data(t_end, dt_mean, stencil_dt, data_save_folder, save_folder, save_name, datalabel,
                 model_labels, l, theta, y_correction, lw, fmt, moments=True, plumes=True, bt = True, two_d=False,
                 zoom_plots=True, n_pores=500, bt_bound_box=None):
    if not stencil_dt:
        stencil_dt = 1e-3
    n_steps = 5
    target_time_array = np.linspace(stencil_dt, t_end, n_steps)[1:]
    target_time_array = np.floor(target_time_array / (10.0 * dt_mean)) * 10.0 * dt_mean
    stencil_labels = model_labels
    t_scale = dt_mean
    l_scale = l
    # binning extents
    xmin = 0.0
    xmax = n_pores * l * np.cos(theta)
    # plot the evolution first and second moment of the plume
    if moments:
        print 'plotting moments...'
        data_name = 'model_moments'
        save_path = os.path.join(data_save_folder, data_name + '.npz')
        model_moments = np.load(save_path)
        com_x_model_array = model_moments['com_x']
        com_y_model_array = model_moments['com_y']
        msd_x_model_array = model_moments['msd_x']
        msd_y_model_array = model_moments['msd_y']
        data_name = 'data_moments'
        save_path = os.path.join(data_save_folder, data_name + '.npz')
        data_moments = np.load(save_path)
        com_x_data = data_moments['com_x']
        com_y_data = data_moments['com_y']
        msd_x_data = data_moments['msd_x']
        msd_y_data = data_moments['msd_y']
        n_points = 12
        print 'calculating plume moments for ' + str(n_points) + ' times'
        target_time_array_data = np.linspace(0.0, np.log(t_end / dt_mean), n_points)
        target_time_array_data = np.hstack((0.0, np.exp(target_time_array_data))) * dt_mean

        target_time_array_model = np.linspace(np.log(stencil_dt / dt_mean), np.log(t_end / dt_mean), n_points)
        target_time_array_model = np.exp(target_time_array_model) * dt_mean



        print 'plotting the moments of the plume...'
        axis_dict = {'ylabel1': 'logitudinal MSD', 'ylabel2': 'logitudinal COM'}
        # axis_dict = {'ylabel1': r"$logitudinal\;MSD$", 'ylabel2': r"$logitudinal\;COM$"}
        save_prefix = 'x_'
        plot_msd_com_both_one(target_time_array_model, com_x_model_array, msd_x_model_array, target_time_array_data,
                              com_x_data, msd_x_data, save_folder, save_name, save_prefix, axis_dict,
                              datalabel, model_labels, t_scale=dt_mean, lw=lw, fmt=fmt)

        axis_dict = {'ylabel1': r"$transverse\;MSD$", 'ylabel2': r"$transverse\;COM$", 'ymin': 400, 'ymax': 600}
        save_prefix = 'y_'
        plot_msd_com_both_one(target_time_array_model, com_y_model_array, msd_y_model_array, target_time_array_data,
                              com_y_data, msd_y_data, save_folder, save_name, save_prefix, axis_dict,
                              datalabel, model_labels, t_scale=dt_mean, lw=lw, fmt=fmt)

        # axis_dict = {'ylabel1': r"$logitudinal\;MSD$", 'ylabel2': r"$transverse\;MSD$"}
        if zoom_plots:
            axis_dict = {'ylabel1': 'logitudinal MSD'}
            save_prefix = 'msd_x'
            # plot_moment_inset(target_time_array_data, msd_x_data, msd_y_data, datalabel, target_time_array_model,
            #               msd_x_model_array, msd_y_model_array, model_labels, t_scale, [], [],
            #               save_folder, axis_dict, lw=lw, fmt=fmt)
            print target_time_array_data/dt_mean
            zoom_box = [[110,200],[30,45]]
            zoom = 3.2
            plot_moment_zoom(target_time_array_data, msd_x_data, datalabel, target_time_array_model,
                             msd_x_model_array, model_labels, t_scale, zoom_box, zoom,
                             save_folder, save_prefix, axis_dict, lw=1, fmt=fmt, legloc=2)

            # axis_dict = {'ylabel1': r"$transverse\;MSD$"}
            axis_dict = {'ylabel1': 'transverse MSD'}
            save_prefix = 'msd_y'
            # zoom_box = [[92, 180], [9, 16]]
            zoom_box = [[110, 200], [9, 16]]
            zoom = 3.2
            plot_moment_zoom(target_time_array_data, msd_y_data, datalabel, target_time_array_model,
                             msd_y_model_array, model_labels, t_scale, zoom_box, zoom,
                             save_folder, save_prefix, axis_dict, lw=1, fmt=fmt, zoomloc=2, cor1=1, cor2=3, legloc=4)
        print 'done'

    if plumes:
        # plot the plume spreading in x and y direction
        n_steps = 5
        target_time_array = np.linspace(stencil_dt, t_end, n_steps)[1:]
        target_time_array = np.floor(target_time_array / (10.0 * dt_mean)) * 10.0 * dt_mean

        data_name = 'data_plumes'
        save_path = os.path.join(data_save_folder, data_name + '.pkl')
        with open(save_path, 'rb') as infile:
            data_plumes = cPickle.load(infile)
        data_plume_x_array, data_plume_y_array = data_plumes[0], data_plumes[1]
        data_name = 'model_plumes'
        save_path = os.path.join(data_save_folder, data_name + '.pkl')
        with open(save_path, 'rb') as infile:
            model_plumes = cPickle.load(infile)
        model_plume_x_array, model_plume_y_array = model_plumes[0], model_plumes[1]
        data_labels = [datalabel]

        print 'plotting the plume spreading in:'
        print 'x direction...'
        nbins = 150
        # plotting extents
        x_min_plot = 0.0
        x_max_plot = xmax / l
        attrib = 'x'
        figsize = [6, 4]
        # plot_plume_evolution_histogram(target_time_array, nbins, xmin, xmax, attrib,
        #                                save_folder, data_plume_x_array, model_plume_x_array, stencil_labels,
        #                                data_labels, save_name, x_min_plot, x_max_plot, l, t_scale=dt_mean, lw=lw,
        #                                fmt=fmt)
        #
        plot_plume_evolution_histogram(target_time_array, nbins, xmin, xmax, attrib,
                                       save_folder, data_plume_x_array, model_plume_x_array, stencil_labels,
                                       data_labels, save_name, x_min_plot, x_max_plot, l, t_scale=dt_mean,
                                       lw=lw, fmt=fmt, figsize=figsize, save_pre='sm')
        zoom = False
        for jj in range(len(target_time_array)):
            if jj == len(target_time_array) - 1:
                if zoom_plots:
                    zoom = True
            plot_plume_evolution_histogram(target_time_array, nbins, xmin, xmax, attrib, save_folder,
                                           data_plume_x_array, model_plume_x_array, stencil_labels, data_labels,
                                           save_name, x_min_plot, x_max_plot, l, t_scale=dt_mean, figsize=figsize,
                                           tidx=jj, lw=lw, fmt=fmt, zoom=zoom)

        print 'y direction...'
        attrib = 'y_array'
        # binning extents
        com_const = y_correction
        if com_const:
            delta = 0.15 * com_const
        else:
            delta = 0.3*xmax
        ymin = (com_const - delta)
        ymax = (com_const + delta)
        nbins = 150
        # plotting extents
        y_min_plot = ymin / l
        y_max_plot = ymax / l
        attrib = 'y'
        # plot_plume_evolution_histogram(target_time_array, nbins, ymin, ymax, attrib,
        #                                save_folder, data_plume_y_array, model_plume_y_array, stencil_labels,
        #                                data_labels, save_name, y_min_plot, y_max_plot, l, t_scale=dt_mean, lw=lw,
        #                                fmt=fmt)
        #
        plot_plume_evolution_histogram(target_time_array, nbins, ymin, ymax, attrib,
                                       save_folder, data_plume_y_array, model_plume_y_array, stencil_labels,
                                       data_labels, save_name, y_min_plot, y_max_plot, l, t_scale=dt_mean,
                                       lw=lw, fmt=fmt, figsize=figsize, save_pre='sm')

        for jj in range(len(target_time_array)):
            plot_plume_evolution_histogram(target_time_array, nbins, ymin, ymax, attrib, save_folder,
                                           data_plume_y_array, model_plume_y_array, stencil_labels, data_labels,
                                           save_name, y_min_plot, y_max_plot, l, t_scale=dt_mean, figsize=figsize,
                                           tidx=jj,
                                           lw=lw, fmt=fmt)
        print 'done...'

        print 'plotting side by side plumes...'
        nxbins = nbins
        nybins = nbins
        for i in range(len(target_time_array)):
            t_target = target_time_array[i]
            model_x_plume_list = []
            model_y_plume_list = []
            data_x_plume = data_plume_x_array[0][i, :]
            data_y_plume = data_plume_y_array[0][i, :]
            for j in range(len(model_plume_x_array)):
                model_x_plume_list.append(model_plume_x_array[j][i, :])
                model_y_plume_list.append(model_plume_y_array[j][i, :])
            plot_plume_x_side_y_oneTime(t_target, nxbins, nybins, xmin, xmax, ymin, ymax, data_x_plume, data_y_plume,
                                        model_x_plume_list, model_y_plume_list, model_labels, t_scale, l_scale,
                                        lw, fmt, save_folder)

        plot_plume_x_side_y(target_time_array, nxbins, nybins, xmin, xmax, ymin, ymax, data_plume_x_array,
                            data_plume_y_array,
                            model_plume_x_array, model_plume_y_array, model_labels, t_scale, l_scale,
                            lw, fmt, save_folder)
        print 'done...'

    if bt:
        print 'plot time to get to a location'
        data_name = 'data_bt'
        save_path = os.path.join(data_save_folder, data_name + '.pkl')
        with open(save_path, 'rb') as infile:
            data_bt_array = cPickle.load(infile)
        data_name = 'model_bt'
        save_path = os.path.join(data_save_folder, data_name + '.pkl')
        with open(save_path, 'rb') as infile:
            model_bt_array = cPickle.load(infile)
        l_frac_array = np.array([0.25, 0.5, 0.75])
        ## for each target length, for all models make the curve
        zoom = False
        for idx_x, target_frac in enumerate(l_frac_array):
            input_array = []
            label_array = [r"$data$"]
            input_array.append(data_bt_array[0][idx_x, :])
            for idx_model in range(len(model_labels)):
                input_array.append(model_bt_array[idx_model][idx_x, :])
                label_array.append(model_labels[idx_model])
            plot_bt(input_array, label_array, dt_mean, target_frac, save_folder, fmt=fmt, lw=lw, zoom=zoom,
                    bound_box=bt_bound_box)
            if idx_x == len(l_frac_array)-1:
                zoom = True
            if zoom_plots:
                plot_bt(input_array, label_array, dt_mean, target_frac, save_folder, fmt=fmt, lw=lw, zoom=zoom)
            # plot_bt_logscale(input_array, label_array, dt_mean, target_frac, save_folder, fmt=fmt, lw=lw)
    if two_d:
        # plot the average plume in 2d
        print 'generating 2d plume figures...'
        #turn of latex rendering, causes issues on the cluster
        # plt.rc('text', usetex=False)
        data_name = 'xy_contour'
        save_path = os.path.join(data_save_folder, data_name + '.npz')
        loader = np.load(save_path)
        X, Y = loader['X'], loader['Y']
        data_name = 'z_contour'
        save_path = os.path.join(data_save_folder, data_name + '.npz')
        loader = np.load(save_path)
        Z, Z2 = loader['dataZ'], loader['modelZ']
        data_name = 'ycorrections'
        save_path = os.path.join(data_save_folder, data_name + '.npz')
        loader = np.load(save_path)
        y_center, dy = loader['y_center'], loader['dy']
        nlevels = 6
        plot_plume_2d_from_saved(target_time_array, nlevels, X, Y, Z, Z2, y_center, dy, save_folder, save_name,
                                 dt_mean, scale_str=r"\overline{\delta t}$", l_scale=1.0, fmt='png')
        print 'done'

def plot_wrapper(t_end, dt_mean, stencil_dt, save_folder, save_name, model_array, data,
                 model_labels, l, theta, y_correction, lw, fmt, moments=True, plumes=True, bt = True, two_d=False):
    """
    :param t_end: final pot time
    :param dt_mean: average jump time from data
    :param stencil_dt: dt used for the stencil
    :param save_folder: main folder to save these plots
    :param save_name: prefix name for saving
    :param model_array:
    :param data:
    :param model_labels:
    :param l:
    :param theta:
    :param y_correction:
    :param lw:
    :param fmt:
    :return:
    """
    t_scale = dt_mean
    l_scale = l
    data.label = r'$data$'
    data_array = [data]
    # binning extents
    xmin = 0.0
    xmax = 500.0 * l * np.cos(theta)
    # plot the evolution first and second moment of the plume
    if moments:
        n_points = 12
        print 'calculating plume moments for ' + str(n_points) + ' times'
        target_time_array_data = np.linspace(0.0, np.log(t_end / dt_mean), n_points)
        target_time_array_data = np.hstack((0.0, np.exp(target_time_array_data))) * dt_mean

        target_time_array_model = np.linspace(np.log(stencil_dt / dt_mean), np.log(t_end / dt_mean), n_points)
        target_time_array_model = np.exp(target_time_array_model) * dt_mean

        com_x_model_array = []
        com_y_model_array = []
        msd_x_model_array = []
        msd_y_model_array = []
        for model in model_array:
            com_x_model, msd_x_model, com_y_model, msd_y_model = plume_msd_com_multiple_times(target_time_array_model,
                                                                                              model)
            com_x_model_array.append(com_x_model)
            com_y_model_array.append(com_y_model)
            msd_x_model_array.append(msd_x_model)
            msd_y_model_array.append(msd_y_model)
        com_x_data, msd_x_data, com_y_data, msd_y_data = plume_msd_com_multiple_times(target_time_array_data, data)

        print 'plotting the moments of the plume...'
        # axis_dict = {'ylabel1': 'logitudinal MSD', 'ylabel2': 'logitudinal COM'}
        axis_dict = {'ylabel1': r"$logitudinal\;MSD$", 'ylabel2': r"$logitudinal\;COM$"}
        save_prefix = 'x_'
        plot_msd_com_both_one(target_time_array_model, com_x_model_array, msd_x_model_array, target_time_array_data,
                              com_x_data, msd_x_data, save_folder, save_name, save_prefix, axis_dict,
                              data.label, model_labels, t_scale=dt_mean, lw=lw, fmt=fmt)

        # axis_dict = {'ylabel1': 'transverse MSD', 'ylabel2': 'transverse COM', 'ymin': 400, 'ymax': 600}
        axis_dict = {'ylabel1': r"$transverse\;MSD$", 'ylabel2': r"$transverse\;COM$", 'ymin': 400, 'ymax': 600}

        save_prefix = 'y_'
        plot_msd_com_both_one(target_time_array_model, com_y_model_array, msd_y_model_array, target_time_array_data,
                              com_y_data, msd_y_data, save_folder, save_name, save_prefix, axis_dict,
                              data.label, model_labels, t_scale=dt_mean, lw=lw, fmt=fmt)
        print 'done'

    if plumes:
        # plot the plume spreading in x and y direction
        n_steps = 5
        target_time_array = np.linspace(stencil_dt, t_end, n_steps)[1:]
        target_time_array = np.floor(target_time_array / (10.0 * dt_mean)) * 10.0 * dt_mean
        data_plume_x_array = []
        data_plume_y_array = []
        data_labels = []
        for data in data_array:
            xtemp, ytemp = plume_location_multiple_times(target_time_array, data.x_array,
                                                         data.y_array, data.t_array)
            data_plume_x_array.append(xtemp)
            data_plume_y_array.append(ytemp)
            data_labels.append(data.label)
        model_plume_x_array = []
        model_plume_y_array = []
        stencil_labels = []
        for model in model_array:
            xtemp, ytemp = plume_location_multiple_times(target_time_array, model.x_array,
                                                         model.y_array, model.t_array)
            model_plume_x_array.append(xtemp)
            model_plume_y_array.append(ytemp)
            stencil_labels.append(model.label)
        del xtemp
        del ytemp


        print 'plotting the plume spreading in:'
        print 'x direction...'
        nbins = 150
        # plotting extents
        x_min_plot = 0.0
        x_max_plot = xmax / l
        attrib = 'x'
        figsize = [6, 4]
        plot_plume_evolution_histogram(target_time_array, nbins, xmin, xmax, attrib,
                                       save_folder, data_plume_x_array, model_plume_x_array, stencil_labels,
                                       data_labels, save_name, x_min_plot, x_max_plot, l, t_scale=dt_mean, lw=lw, fmt=fmt)

        plot_plume_evolution_histogram(target_time_array, nbins, xmin, xmax, attrib,
                                       save_folder, data_plume_x_array, model_plume_x_array, stencil_labels,
                                       data_labels, save_name, x_min_plot, x_max_plot, l, t_scale=dt_mean,
                                       lw=lw, fmt=fmt, figsize=figsize, save_pre='sm')

        for jj in range(len(target_time_array)):
            plot_plume_evolution_histogram(target_time_array, nbins, xmin, xmax, attrib, save_folder,
                                           data_plume_x_array, model_plume_x_array, stencil_labels, data_labels,
                                           save_name, x_min_plot, x_max_plot, l, t_scale=dt_mean, figsize=figsize, tidx=jj,
                                           lw=lw, fmt=fmt)

        print 'y direction...'
        attrib = 'y_array'
        # binning extents
        com_const = y_correction
        delta = 0.15 * com_const
        ymin = (com_const - delta)
        ymax = (com_const + delta)
        nbins = 150
        # plotting extents
        y_min_plot = ymin / l
        y_max_plot = ymax / l
        attrib = 'y'
        plot_plume_evolution_histogram(target_time_array, nbins, ymin, ymax, attrib,
                                       save_folder, data_plume_y_array, model_plume_y_array, stencil_labels,
                                       data_labels, save_name, y_min_plot, y_max_plot, l, t_scale=dt_mean, lw=lw, fmt=fmt)

        plot_plume_evolution_histogram(target_time_array, nbins, ymin, ymax, attrib,
                                       save_folder, data_plume_y_array, model_plume_y_array, stencil_labels,
                                       data_labels, save_name, y_min_plot, y_max_plot, l, t_scale=dt_mean,
                                       lw=lw, fmt=fmt, figsize=figsize, save_pre='sm')

        for jj in range(len(target_time_array)):
            plot_plume_evolution_histogram(target_time_array, nbins, ymin, ymax, attrib, save_folder,
                                           data_plume_y_array, model_plume_y_array, stencil_labels, data_labels,
                                           save_name, y_min_plot, y_max_plot, l, t_scale=dt_mean, figsize=figsize, tidx=jj,
                                           lw=lw, fmt=fmt)
        print 'done...'

        print 'plotting side by side plumes...'
        nxbins = nbins
        nybins = nbins
        for i in range(len(target_time_array)):
            t_target = target_time_array[i]
            model_x_plume_list = []
            model_y_plume_list = []
            data_x_plume = data_plume_x_array[0][i, :]
            data_y_plume = data_plume_y_array[0][i, :]
            for j in range(len(model_plume_x_array)):
                model_x_plume_list.append(model_plume_x_array[j][i, :])
                model_y_plume_list.append(model_plume_y_array[j][i, :])
            plot_plume_x_side_y_oneTime(t_target, nxbins, nybins, xmin, xmax, ymin, ymax, data_x_plume, data_y_plume,
                                model_x_plume_list, model_y_plume_list, model_labels, t_scale, l_scale,
                                lw, fmt, save_folder)

        plot_plume_x_side_y(target_time_array, nxbins, nybins, xmin, xmax, ymin, ymax, data_plume_x_array, data_plume_y_array,
                            model_plume_x_array, model_plume_y_array, model_labels, t_scale, l_scale,
                            lw, fmt, save_folder)
        print 'done...'

    if bt:
        print 'plot time to get to a location'
        l_frac_array  = np.array([0.25, 0.5, 0.75])
        target_x_array = l_frac_array*xmax
        data_bt_array = []
        data_labels = []
        for data in data_array:
            ttemp = plume_bt_multiple_locations(target_x_array, data.x_array,
                                                       data.t_array)
            data_bt_array.append(ttemp)
            data_labels.append(data.label)
        model_bt_array = []
        stencil_labels = []
        for model in model_array:
            ttemp = plume_bt_multiple_locations(target_x_array, model.x_array,
                                                model.t_array)
            model_bt_array.append(ttemp)
            stencil_labels.append(model.label)
        ## for each target length, for all models make the curve
        for idx_x, target_frac in enumerate(l_frac_array):
            input_array = []
            label_array = [r"$data$"]
            input_array.append(data_bt_array[0][idx_x, :])
            for idx_model in range(len(model_array)):
                input_array.append(model_bt_array[idx_model][idx_x, :])
                label_array.append(model_labels[idx_model])
            plot_bt(input_array, label_array, dt_mean, target_frac, save_folder, fmt=fmt, lw=lw)
            # plot_bt_logscale(input_array, label_array, dt_mean, target_frac, save_folder, fmt=fmt, lw=lw)
    if two_d:
        #plot the average plume in 2d
        print 'generating 2d plume figures...'
        plt.rc('text', usetex=False)
        nlevels = 6
        n_query = 100j
        save_folder_gif = os.path.join(save_folder, 'gif_pics')
        if not os.path.exists(save_folder_gif):
            os.mkdir(save_folder_gif)
        plot_plume_2d_with_kde(target_time_array, nlevels, n_query, model, data, save_folder_gif, save_name,
                               t_scale=dt_mean, max_samples=400000, l_scale=l)
        print 'done'