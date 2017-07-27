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
from copy import copy
from py_dp.dispersion.binning import class_index_log_log
from py_dp.dispersion.binning import class_index_abs_log
from py_dp.dispersion.binning import fix_out_of_bound
import os
import pickle
from py_dp.dispersion.convert_to_time_process_with_freq import get_time_dx_dy_array_with_freq, remove_duplicate_xy
# from py_dp.dispersion.trajectory_count_cython import fill_one_trajectory_sparse_cython
# from scipy.sparse import csc_matrix
#
# from py_dp.dispersion.trajectory_count_cython import count_with_lag_one_trajectory_cython

def count_matrix_with_lag(lag, nClass, v_array, freq_array, pointer_list,
                          v_neg_log_edges, v_pos_log_edges):
    """
    This is the count matrix in time using my bins (different classes for positive and negative velocities)
    """
    #first test the bins
    v_pos_max = np.amax(v_array[v_array > 0.0])
    assert(v_pos_max < np.exp(v_pos_log_edges[-1]))
    transition_count_matrix = np.zeros((nClass,nClass))
    start_idx = 0
    for i in pointer_list:
        v_temp = v_array[start_idx:i]
        freq_temp = freq_array[start_idx:i]
        start_idx = i
        inds_temp = class_index_log_log(v_temp,v_neg_log_edges,v_pos_log_edges)
        idx_cum1 = 1
        idx_cum2 = 1 + lag
        cumsum_temp = np.cumsum(freq_temp)
        last_index = cumsum_temp[-1]
        while idx_cum2 < last_index:
            [idx1, idx2] = np.digitize([idx_cum1, idx_cum2], cumsum_temp, right=True)
            if idx1 != idx2:
                #in this case they are not in the same cumsum_bin hence no repetition
                #current bin is idx1
                transition_count_matrix[inds_temp[idx2], inds_temp[idx1]] += 1
                idx_cum1 += 1
                idx_cum2 += 1
            else:
                #they are in a bin with repetition might happen
                n_repeating_pair = (cumsum_temp[idx1] - idx_cum2) + 1
                current_bin = inds_temp[idx1]
                transition_count_matrix[current_bin, current_bin] += n_repeating_pair
                idx_cum1 += n_repeating_pair
                idx_cum2 += n_repeating_pair
    return transition_count_matrix

def count_matrix_with_freq_one_trajectory_kang(transition_count_matrix, lag, v_array, freq_array, v_log_edges):
    """
    Count matrix in time using kang bins
    """
    if not len(v_array):
        return
    inds_temp = class_index_abs_log(v_array, v_log_edges)
    idx_cum1 = 1
    idx_cum2 = 1 + lag
    cumsum_temp = np.cumsum(freq_array)
    last_index = cumsum_temp[-1]
    while idx_cum2 < last_index:
        [idx1, idx2] = np.digitize([idx_cum1, idx_cum2], cumsum_temp, right=True)
        if idx1 != idx2:
            #in this case they are not in the same cumsum_bin hence no repetition
            #current bin is idx1
            transition_count_matrix[inds_temp[idx2], inds_temp[idx1]] += 1
            idx_cum1 += 1
            idx_cum2 += 1
        else:
            #they are in a bin with repetition might happen
            n_repeating_pair = (cumsum_temp[idx1] - idx_cum2) + 1
            current_bin = inds_temp[idx1]
            transition_count_matrix[current_bin, current_bin] += n_repeating_pair
            idx_cum1 += n_repeating_pair
            idx_cum2 += n_repeating_pair

def count_matrix_with_freq_one_trajectory_agg(transition_count_matrix, lag, v_array, freq_array, v_log_edges):
    """
    Count matrix in time using kang bins
    """
    if not len(v_array):
        return
    inds_temp = np.digitize(np.log(np.abs(v_array)), v_log_edges)
    fix_out_of_bound(inds_temp, v_log_edges)
    inds_temp -= 1
    idx_cum1 = 1
    idx_cum2 = 1 + lag
    cumsum_temp = np.cumsum(freq_array)
    last_index = cumsum_temp[-1]
    while idx_cum2 < last_index:
        [idx1, idx2] = np.digitize([idx_cum1, idx_cum2], cumsum_temp, right=True)
        if idx1 != idx2:
            #in this case they are not in the same cumsum_bin hence no repetition
            #current bin is idx1
            transition_count_matrix[inds_temp[idx2], inds_temp[idx1]] += 1
            idx_cum1 += 1
            idx_cum2 += 1
        else:
            #they are in a bin with repetition might happen
            n_repeating_pair = (cumsum_temp[idx1] - idx_cum2) + 1
            current_bin = inds_temp[idx1]
            transition_count_matrix[current_bin, current_bin] += n_repeating_pair
            idx_cum1 += n_repeating_pair
            idx_cum2 += n_repeating_pair


def count_matrix_with_freq_one_trajectory(transition_count_matrix, lag, ind_array, freq_array):
    """
    Count matrix in time using kang bins
    """
    if not len(ind_array):
        return
    idx_cum1 = 1
    idx_cum2 = 1 + lag
    cumsum_temp = np.cumsum(freq_array)
    last_index = cumsum_temp[-1]
    while idx_cum2 < last_index:
        [idx1, idx2] = np.digitize([idx_cum1, idx_cum2], cumsum_temp, right=True)
        if idx1 != idx2:
            #in this case they are not in the same cumsum_bin hence no repetition
            #current bin is idx1
            transition_count_matrix[ind_array[idx2], ind_array[idx1]] += 1
            idx_cum1 += 1
            idx_cum2 += 1
        else:
            #they are in a bin with repetition might happen
            n_repeating_pair = (cumsum_temp[idx1] - idx_cum2) + 1
            current_bin = ind_array[idx1]
            transition_count_matrix[current_bin, current_bin] += n_repeating_pair
            idx_cum1 += n_repeating_pair
            idx_cum2 += n_repeating_pair

def count_matrix_one_trajectory(trans_count_matrix, lag, ind_array):
    """
    count the transitions (no frequency)
    :param trans_count_matrix: transition count matrix
    :param lag: lag
    :param ind_array: indices observed in this path (process trajectory)
    :return:
    """
    if not len(ind_array):
        return
    for j in range(len(ind_array) - lag):
        nextBin = ind_array[j + lag]
        currentBin = ind_array[j]
        trans_count_matrix[nextBin, currentBin] += 1

def normalize_columns(countMatrix):
    tt = copy(countMatrix)
    nClass = countMatrix.shape[0]
    for i in range(nClass):
        colSum = np.sum(tt[:, i])
        if colSum != 0:
            tt[:, i] = tt[:, i] / float(colSum)
    return tt


def get_trans_matrix_single_attrib(lag_array, n_realz, input_folder, mapping, time_step, prefix='real_',
                                   numbered=True, verbose=False):
    if (not numbered) and n_realz>1:
        raise 'Expecting only one file when no numbers are used for the input data'
    v_log_edges = mapping.v_log_edges
    n_v_class = mapping.n_abs_v_classes
    n_theta_class = mapping.n_theta_classes
    theta_edges = mapping.theta_edges
    v_output_list = []
    theta_output_list = []
    for lag in lag_array:
        print " extracting matrices for lag = ", lag
        v_count_matrix = np.zeros((n_v_class, n_v_class))
        t_count_matrix = np.zeros((n_theta_class, n_theta_class))
        for j in range(n_realz):
            if verbose and not j%20:
                print 'realziation ', j
            if numbered:
                file_name = prefix + str(j) + ".pkl"
            else:
                file_name = prefix + ".pkl"
            input_file = os.path.join(input_folder, file_name)
            with open(input_file, 'rb') as input:
                dataHolder = pickle.load(input)
            dx = np.diff(dataHolder.x_array)
            dy = np.diff(dataHolder.y_array)
            dt = np.diff(dataHolder.t_array)
            if not (dx.shape[0] and dy.shape[0] and dt.shape[0]):
                print 'some array was empty, skipping this file...'
                continue
            lastIdx = dataHolder.last_idx_array
            vxMatrix = np.divide(dx, dt)
            vyMatrix = np.divide(dy, dt)
            m = dx.shape[0]
            for i in range(m):
                x_start = dataHolder.x_array[i, 0]
                y_start = dataHolder.y_array[i, 0]
                # get the time process for each velocity
                cutOff = lastIdx[i]
                dxTime, dyTime, freq = get_time_dx_dy_array_with_freq(dt[i, :cutOff], vxMatrix[i, :cutOff],
                                                                      vyMatrix[i, :cutOff], x_start, y_start,
                                                                      time_step)
                v_temp = np.sqrt(np.power(dxTime, 2) + np.power(dyTime, 2)) / time_step
                theta_temp = np.arctan2(dyTime, dxTime)
                if len(v_temp) > lag:
                    new_v, new_theta, new_f = remove_duplicate_xy(v_temp, theta_temp, freq)
                    class_v = np.array(mapping.find_1d_class_idx(np.log(new_v), v_log_edges), dtype=int)
                    class_theta = np.array(mapping.find_1d_class_idx(new_theta, theta_edges), dtype=int)
                    count_matrix_with_freq_one_trajectory(v_count_matrix, lag, class_v, new_f)
                    count_matrix_with_freq_one_trajectory(t_count_matrix, lag, class_theta, new_f)
        v_output_list.append(v_count_matrix)
        theta_output_list.append(t_count_matrix)
    return v_output_list, theta_output_list


def get_trans_matrix_single_attrib_both_methods(lag_array, n_realz, input_folder, mapping, time_step, prefix='real_',
                                                numbered=True, verbose=False, average_available=False):
    if average_available:
        return get_trans_matrix_single_attrib_both_methods_from_avg(lag_array, n_realz, input_folder, mapping,
                                                                    prefix=prefix, numbered=numbered,
                                                                    verbose=verbose)
    else:
        return get_trans_matrix_single_attrib_both_methods_from_scratch(lag_array, n_realz, input_folder, mapping,
                                                                        time_step, prefix=prefix, numbered=numbered,
                                                                        verbose=verbose)

def get_trans_matrix_single_attrib_both_methods_from_avg(lag_array, n_realz, input_folder, mapping,
                                                             prefix='real_', numbered=True, verbose=False):
    """
    Get the aggregate transition matrix both considering the frequency and not considering the frequency
    corresponding to the stencil method and the extended stencil method
    :param lag_array:
    :param n_realz:
    :param input_folder:
    :param mapping:
    :param time_step:
    :param prefix:
    :param numbered:
    :param verbose:
    :return:
    """
    if (not numbered) and n_realz>1:
        raise 'Expecting only one file when no numbers are used for the input data'
    v_log_edges = mapping.v_log_edges
    n_v_class = mapping.n_abs_v_classes
    n_theta_class = mapping.n_theta_classes
    theta_edges = mapping.theta_edges
    v_output_list = [np.zeros((n_v_class, n_v_class)) for i in range(2)]
    theta_output_list = [np.zeros((n_theta_class, n_theta_class)) for i in range(2)]
    v_output_list_nofreq = [np.zeros((n_v_class, n_v_class)) for i in range(2)]
    theta_output_list_nofreq = [np.zeros((n_theta_class, n_theta_class)) for i in range(2)]
    start_idx = 0
    for j in range(n_realz):
        # load the polar coordinates file
        data_path = os.path.join(input_folder, 'avg_polar_' + str(j) + '.npz')
        data = np.load(data_path)
        big_v, big_theta, big_f, ptr_list = data['V'], data['Theta'], data['F'], data['ptr']
        for i in ptr_list:
            new_v, new_theta, new_f = big_v[start_idx:i], big_theta[start_idx:i], big_f[start_idx:i]
            start_idx = i
            for idx_lag, lag in enumerate(lag_array):
                if len(new_v) > lag:
                    class_v = np.array(mapping.find_1d_class_idx(np.log(new_v), v_log_edges), dtype=int)
                    class_theta = np.array(mapping.find_1d_class_idx(new_theta, theta_edges), dtype=int)
                    count_matrix_with_freq_one_trajectory(v_output_list[idx_lag], lag, class_v, new_f)
                    count_matrix_with_freq_one_trajectory(theta_output_list[idx_lag], lag, class_theta, new_f)
                    # get the transition matrices for the extended method (v, theta, f) ->
                    # input (v,theta)
                    count_matrix_one_trajectory(v_output_list_nofreq[idx_lag], lag, class_v)
                    count_matrix_one_trajectory(theta_output_list_nofreq[idx_lag], lag, class_theta)
    return v_output_list, theta_output_list, v_output_list_nofreq, theta_output_list_nofreq



def get_trans_matrix_single_attrib_both_methods_from_scratch(lag_array, n_realz, input_folder, mapping, time_step,
                                                             prefix='real_', numbered=True, verbose=False):
    """
    Get the aggregate transition matrix both considering the frequency and not considering the frequency
    corresponding to the stencil method and the extended stencil method
    :param lag_array:
    :param n_realz:
    :param input_folder:
    :param mapping:
    :param time_step:
    :param prefix:
    :param numbered:
    :param verbose:
    :return:
    """
    if (not numbered) and n_realz>1:
        raise 'Expecting only one file when no numbers are used for the input data'
    v_log_edges = mapping.v_log_edges
    n_v_class = mapping.n_abs_v_classes
    n_theta_class = mapping.n_theta_classes
    theta_edges = mapping.theta_edges
    v_output_list = [np.zeros((n_v_class, n_v_class)) for i in range(2)]
    theta_output_list = [np.zeros((n_theta_class, n_theta_class)) for i in range(2)]
    v_output_list_nofreq = [np.zeros((n_v_class, n_v_class)) for i in range(2)]
    theta_output_list_nofreq = [np.zeros((n_theta_class, n_theta_class)) for i in range(2)]
    for j in range(n_realz):
        if verbose and not j%20:
            print 'realziation ', j
        if numbered:
            file_name = prefix + str(j) + ".pkl"
        else:
            file_name = prefix + ".pkl"
        input_file = os.path.join(input_folder, file_name)
        with open(input_file, 'rb') as input:
            dataHolder = pickle.load(input)
        dx = np.diff(dataHolder.x_array)
        dy = np.diff(dataHolder.y_array)
        dt = np.diff(dataHolder.t_array) + 1e-12
        if not (dx.shape[0] and dy.shape[0] and dt.shape[0]):
            print 'some array was empty, skipping this file...'
            continue
        lastIdx = dataHolder.last_idx_array
        vxMatrix = np.divide(dx, dt)
        vyMatrix = np.divide(dy, dt)
        m = dx.shape[0]
        for i in range(m):
            x_start = dataHolder.x_array[i, 0]
            y_start = dataHolder.y_array[i, 0]
            # get the time process for each velocity
            cutOff = lastIdx[i]
            dxTime, dyTime, freq = get_time_dx_dy_array_with_freq(dt[i, :cutOff], vxMatrix[i, :cutOff],
                                                                  vyMatrix[i, :cutOff], x_start, y_start,
                                                                  time_step)
            v_temp = np.sqrt(np.power(dxTime, 2) + np.power(dyTime, 2)) / time_step
            theta_temp = np.arctan2(dyTime, dxTime)
            new_v, new_theta, new_f = remove_duplicate_xy(v_temp, theta_temp, freq)
            for idx_lag, lag in enumerate(lag_array):
                if len(new_v) > lag:
                    class_v = np.array(mapping.find_1d_class_idx(np.log(new_v), v_log_edges), dtype=int)
                    class_theta = np.array(mapping.find_1d_class_idx(new_theta, theta_edges), dtype=int)
                    count_matrix_with_freq_one_trajectory(v_output_list[idx_lag], lag, class_v, new_f)
                    count_matrix_with_freq_one_trajectory(theta_output_list[idx_lag], lag, class_theta, new_f)
                    # get the transition matrices for the extended method (v, theta, f) ->
                    # input (v,theta)
                    count_matrix_one_trajectory(v_output_list_nofreq[idx_lag], lag, class_v)
                    count_matrix_one_trajectory(theta_output_list_nofreq[idx_lag], lag, class_theta)
    return v_output_list, theta_output_list, v_output_list_nofreq, theta_output_list_nofreq

def get_trans_matrix_v_spatial(lag_array, n_realz, input_folder, mapping,
                                      numbered=True, verbose=False):
    """
    Get the aggregate transition matrix both considering the frequency and not considering the frequency
    corresponding to the stencil method and the extended stencil method
    :param lag_array:
    :param n_realz:
    :param input_folder:
    :param mapping:
    :param time_step:
    :param prefix:
    :param numbered:
    :param verbose:
    :return:
    """
    if (not numbered) and n_realz>1:
        raise 'Expecting only one file when no numbers are used for the input data'
    v_log_edges = mapping.v_log_edges
    n_v_class = mapping.n_abs_v_classes
    n_theta_class = mapping.n_theta_classes
    theta_edges = mapping.theta_edges
    v_output_list = [np.zeros((n_v_class, n_v_class)) for i in range(2)]
    for j in range(n_realz):
        start_idx = 0
        # load the polar coordinates file
        data_path = os.path.join(input_folder, 'polar_'+str(j)+'.npz')
        data = np.load(data_path)
        big_v, big_theta, ptr_list = data['V'], data['Theta'], data['ptr']
        for i in ptr_list:
            new_v, new_theta = big_v[start_idx:i], big_theta[start_idx:i]
            start_idx = i
            for idx_lag, lag in enumerate(lag_array):
                if len(new_v) > lag:
                    class_v = np.array(mapping.find_1d_class_idx(np.log(new_v), v_log_edges), dtype=int)
                    count_matrix_one_trajectory(v_output_list[idx_lag], lag, class_v)
    return v_output_list

# def count_with_lag_one_trajectory(transition_count_matrix, lag, v_array, v_neg_log_edges, v_pos_log_edges ):
#     inds_temp = class_index_log_log(v_array,v_neg_log_edges,v_pos_log_edges)
#     for j in range(len(v_array) - lag):
#         nextBin = inds_temp[j+lag]
#         currentBin = inds_temp[j]
#         transition_count_matrix[nextBin,currentBin] += 1

# def count_matrix_with_lag_without_freq(lag, nClass, v_array, pointer_list,
#                                        v_neg_log_edges, v_pos_log_edges):
#     #first test the bins
#     v_pos_max = np.amax(v_array[v_array > 0.0])
#     assert(v_pos_max < np.exp(v_pos_log_edges[-1]))
#     transition_count_matrix = np.zeros((nClass,nClass))
#     start_idx = 0
#     for i in pointer_list:
#         v_temp = v_array[start_idx:i]
#         start_idx = i
#         count_with_lag_one_trajectory_cython(transition_count_matrix, lag, v_temp, v_neg_log_edges, v_pos_log_edges )
#         # inds_temp = class_index_log_log(v_temp,v_neg_log_edges,v_pos_log_edges)
#         # for j in range(len(v_temp) - lag):
#         #         nextBin = inds_temp[j+lag]
#         #         currentBin = inds_temp[j]
#         #         transition_count_matrix[nextBin,currentBin] += 1
#     return transition_count_matrix

# def count_matrix_with_lag_without_freq_from_file(lag, nClass, folder, n_realization,
#                                        v_neg_log_edges, v_pos_log_edges):
#     #first test the bins
#     v_pos_max = np.amax(v_array[v_array > 0.0])
#     assert(v_pos_max < np.exp(v_pos_log_edges[-1]))
#     transition_count_matrix = np.zeros((nClass,nClass))
#     start_idx = 0
#     for i in pointer_list:
#         v_temp = v_array[start_idx:i]
#         start_idx = i
#         inds_temp = class_index_log_log(v_temp,v_neg_log_edges,v_pos_log_edges)
#         for j in range(len(v_temp) - lag):
#                 nextBin = inds_temp[j+lag]
#                 currentBin = inds_temp[j]
#                 transition_count_matrix[nextBin,currentBin] += 1
#     return transition_count_matrix

