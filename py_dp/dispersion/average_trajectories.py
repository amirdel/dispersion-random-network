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
import pickle
from py_dp.dispersion.convert_to_time_process_with_freq import get_time_dx_dy_array_with_freq
from convert_to_time_process_with_freq import remove_duplicate_xy
import itertools as itertools
from py_dp.dispersion.binning import flatten

def average_all_realizations(input_folder, n_realizations, time_step, save_folder, n_combine=None, prefix='real',
                             verbose=True, print_every=20):
    """
    save averaged dx, dy, freq for given dt
    save v, theta, freq for given dt
    all the things needed for creating bins
    save big_v, big_theta, big_f, big_y
    save init_v, init_theta, init_f
    :param input_folder: folder containing the input realizations
    :param n_realizations: total number of realizations
    :param time_step: time step size
    :param save_folder: full path to folder to save the averaged realizations
    :param n_combine: number of realizations ro combine in each output file
    :param prefix: prefix for input files
    :param verbose: whether to write output messages or not
    :param print_every: output messages print frequency
    """
    if verbose:
        print "averaging realizations..."
    if not n_combine:
        print "n_combine == None --> saving all trajectories in one file..."
        n_combine = n_realizations
    # make folder for saving averaged realizations
    total_length = 0
    # count number of output files
    counter = 0
    # count realizations per output file
    realz_count = 0
    # each realization has 1000 particles
    pointer_list = []
    initial_v = []
    initial_f = []
    initial_theta = []
    big_dx_list, big_dy_list, big_freq_list = [[] for i in range(3)]
    big_v_list, big_theta_list, big_y_list = [[] for i in range(3)]
    # for each realization
    for j in range(n_realizations):
        if verbose and not j%print_every:
            print "reading realization nr ", j
        case_name = prefix + "_" + str(j) + ".pkl"
        input_file = os.path.join(input_folder, case_name)
        with open(input_file, 'rb') as input:
            dataHolder = pickle.load(input)
        dx = np.diff(dataHolder.x_array)
        dy = np.diff(dataHolder.y_array)
        dt = np.diff(dataHolder.t_array) + 1e-15
        lastIdx = dataHolder.last_idx_array
        vxMatrix = np.divide(dx, dt)
        vyMatrix = np.divide(dy, dt)
        m = dx.shape[0]
        # read all the trajectories in this realization
        for i in range(m):
            x_start = dataHolder.x_array[i,0]
            y_start = dataHolder.y_array[i,0]
            # get the time process for each velocity
            cutOff = lastIdx[i]
            dx_time, dy_time, freq = get_time_dx_dy_array_with_freq(dt[i, :cutOff], vxMatrix[i, :cutOff],
                                                                  vyMatrix[i, :cutOff], x_start, y_start, time_step)
            if len(dx_time) < 1:
                continue
            dx_time, dy_time, freq = remove_duplicate_xy(dx_time, dy_time, freq)
            current_v = np.sqrt(np.power(dx_time, 2) + np.power(dy_time, 2))
            current_theta = np.arctan2(dy_time, dx_time)
            current_y = np.cumsum(dy_time)
            current_length = len(dx_time)
            if current_length > 1:
                total_length += current_length
                big_dx_list.append(dx_time)
                big_dy_list.append(dy_time)
                big_v_list.append(current_v)
                big_theta_list.append(current_theta)
                big_y_list.append(current_y)
                big_freq_list.append(freq)
                pointer_list.append(total_length)
                # save the first velocity for initialization
                initial_v.append(current_v[0] / time_step)
                initial_theta.append(current_theta[0])
                initial_f.append(freq[0])
        realz_count += 1
        if n_combine == 1 or (j > 0 and (j + 1) % n_combine == 0) or j + 1 == n_realizations:
            if verbose:
                print '     -saving combined realizations'
            # save this batch and initialize the arrays for next batch
            # flatten the big lists
            chain = itertools.chain(*big_dx_list)
            big_dx_array = np.array(list(chain), dtype=np.float)
            chain = itertools.chain(*big_dy_list)
            big_dy_array = np.array(list(chain), dtype=np.float)
            chain = itertools.chain(*big_freq_list)
            big_freq_array = np.array(list(chain), dtype=np.int)
            chain = itertools.chain(*big_y_list)
            big_y_array = np.array(list(chain), dtype=np.float)
            # save these n_combine averaged realizations in cartesian frame
            save_path = os.path.join(save_folder,'avg_cartesian_'+str(counter)+'.npz')
            np.savez(save_path, DX=big_dx_array, DY=big_dy_array, F=big_freq_array,
                     Y = big_y_array, ptr=pointer_list, dt=time_step, n_realz=realz_count)
            big_dx_list, big_dy_list = [[] for i in range(2)]
            del big_dx_array, big_dy_array
            # save these n_combine averaged realizations in polar coordinates
            chain = itertools.chain(*big_v_list)
            big_v_array = np.array(list(chain), dtype=np.float) / time_step
            chain = itertools.chain(*big_theta_list)
            big_theta_array = np.array(list(chain), dtype=np.float)
            save_path = os.path.join(save_folder, 'avg_polar_'+str(counter)+'.npz')
            np.savez(save_path, V=big_v_array, Theta=big_theta_array, F=big_freq_array,
                     ptr=pointer_list, dt=time_step, n_realz=realz_count)
            big_v_list, big_theta_list, big_freq_list, big_y_list = [[] for i in range(4)]
            # reset pointer array
            pointer_list = []
            total_length = 0
            del big_v_array, big_theta_array, big_freq_array, big_y_array
            counter += 1
            realz_count = 0
    initial_v = np.array(initial_v)
    initial_f = np.array(initial_f, dtype=np.int)
    initial_theta = np.array(initial_theta)
    # save the initial values for v, theta, f
    save_path = os.path.join(save_folder, 'initial_arrays.npz')
    np.savez(save_path, v=initial_v, theta=initial_theta, f=initial_f, dt=time_step)
    # save number of averaged realization files
    save_path = os.path.join(save_folder, 'case_info.npz')
    np.savez(save_path, n_out=counter, n_input=n_realizations, dt=time_step)

def combine_all_realizations(input_folder, n_realizations, save_folder, n_combine=None, prefix='real',
                             verbose=True, print_every=20):
    """
    preprocess structured network spatial data
    save averaged dx, dy, for a given structured network
    save v, theta, for every jump
    all the things needed for creating bins
    save big_v, big_theta, big_y
    save init_v, init_theta
    :param input_folder: folder containing the input realizations
    :param n_realizations: total number of realizations
    :param time_step: time step size
    :param save_folder: full path to folder to save the averaged realizations
    :param n_combine: number of realizations ro combine in each output file
    :param prefix: prefix for input files
    :param verbose: whether to write output messages or not
    :param print_every: output messages print frequency
    """
    if verbose:
        print "averaging realizations..."
    if not n_combine:
        print "n_combine == None --> saving all trajectories in one file..."
        n_combine = n_realizations
    # make folder for saving averaged realizations
    total_length = 0
    # count number of output files
    counter = 0
    # count realizations per output file
    realz_count = 0
    # each realization has 1000 particles
    pointer_list = []
    initial_v = []
    initial_theta = []
    big_dx_list, big_dy_list = [[] for i in range(2)]
    big_v_list, big_theta_list = [[] for i in range(2)]
    # for each realization
    for j in range(n_realizations):
        if verbose and not j%print_every:
            print "reading realization nr ", j
        case_name = prefix + "_" + str(j) + ".pkl"
        input_file = os.path.join(input_folder, case_name)
        with open(input_file, 'rb') as input:
            dataHolder = pickle.load(input)
        dx = np.diff(dataHolder.x_array)
        dy = np.diff(dataHolder.y_array)
        dt = np.diff(dataHolder.t_array) + 1e-15
        last_idx = dataHolder.last_idx_array
        vx_matrix = np.divide(dx, dt)
        vy_matrix = np.divide(dy, dt)
        m = dx.shape[0]
        # read all the trajectories in this realization
        for i in range(m):
            cut_off = last_idx[i]
            current_dx = dx[i, :cut_off]
            current_dy = dy[i, :cut_off]
            # get the time process for each velocity
            vx, vy = vx_matrix[i, :cut_off], vy_matrix[i, :cut_off]
            current_v = np.sqrt(np.power(vx, 2) + np.power(vy, 2))
            current_theta = np.arctan2(vy, vx)
            current_length = len(current_v)
            if current_length > 1:
                total_length += current_length
                big_dx_list.append(current_dx)
                big_dy_list.append(current_dy)
                big_v_list.append(current_v)
                big_theta_list.append(current_theta)
                pointer_list.append(total_length)
                # save the first velocity for initialization
                initial_v.append(current_v[0])
                initial_theta.append(current_theta[0])
        realz_count += 1
        if n_combine == 1 or (j > 0 and (j + 1) % n_combine == 0) or j + 1 == n_realizations:
            if verbose:
                print '     -saving combined realizations'
            # save this batch and initialize the arrays for next batch
            # flatten the big lists
            big_dx_array, big_dy_array = [flatten(ll) for ll in [big_dx_list, big_dy_list]]
            # save these n_combine averaged realizations in cartesian frame
            save_path = os.path.join(save_folder,'cartesian_'+str(counter)+'.npz')
            np.savez(save_path, X=big_dx_array, Y=big_dy_array, ptr=pointer_list, n_realz=realz_count)
            big_dx_list, big_dy_list = [[] for i in range(2)]
            del big_dx_array, big_dy_array
            # save these n_combine averaged realizations in polar coordinates
            big_v_array, big_theta_array = [flatten(ll) for ll in [big_v_list, big_theta_list]]
            save_path = os.path.join(save_folder, 'polar_'+str(counter)+'.npz')
            np.savez(save_path, V=big_v_array, Theta=big_theta_array, ptr=pointer_list, n_realz=realz_count)
            big_v_list, big_theta_list = [[] for i in range(2)]
            # reset pointer array
            pointer_list = []
            total_length = 0
            del big_v_array, big_theta_array
            counter += 1
            realz_count = 0
    initial_v = np.array(initial_v)
    initial_theta = np.array(initial_theta)
    # save the initial values for v, theta, f
    save_path = os.path.join(save_folder, 'initial_arrays.npz')
    np.savez(save_path, v=initial_v, theta=initial_theta)
    # save number of averaged realization files
    save_path = os.path.join(save_folder, 'case_info.npz')
    np.savez(save_path, n_out=counter, n_input=n_realizations)