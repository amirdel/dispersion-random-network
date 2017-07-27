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
from copy import copy
import bisect as bs
from py_dp.dispersion.convert_to_time_process_with_freq import get_time_dx_array_with_frequency, get_time_dx_dy_array_with_freq
from convert_to_time_process_with_freq import remove_duplicate_xy
import itertools as itertools


def get_cdf(input1):
    # sum of array should be one
    input_array = copy(input1)
    input_array = input_array / float(np.sum(input_array))
    cdf = [input_array[0]]
    for i in xrange(1, len(input_array)):
        cdf.append(cdf[-1] + input_array[i])
    return np.array(cdf, dtype=np.float)

def get_cdf_from_bins(input_array, input_bins):
    sort_idx = np.argsort(input_array)
    sorted_input = input_array[sort_idx]
    #histogram of the values that occur ignoring the frequency
    h,bins = np.histogram(sorted_input, bins = input_bins)
    center_vals = 0.5*np.diff(bins) + bins[:-1]
    cdf = get_cdf(h)
    return center_vals, cdf

def class_index_log_log(inputArray, negEdges, PosEdges):
    """
    input: array of velocities, edges for negative velocity bins, edges
    for positive velocity bins.
    output: index (starting from 0) in the velocity bins
    important check: min(negEdges) < inputArray < max(posEdges)
    """
    len_input = len(inputArray)
    idxArray = -5 * np.ones(len_input, dtype=np.int)
    n_negative_classes = len(negEdges) - 1
    posMask = (inputArray > 0.0)
    negMask = ~posMask
    n_neg_value = len(np.where(negMask)[0])
    # find class for negative entries
    if n_neg_value > 0:
        negIdx = np.digitize(np.log(-inputArray[negMask]), negEdges)
        fix_out_of_bound(negIdx, negEdges)
        idxArray[negMask] = negIdx - 1
    # find class for positive entries
    # these are logarithmic classes
    if n_neg_value != len_input:
        posIdx = np.digitize(np.log(inputArray[posMask]), PosEdges)
        fix_out_of_bound(posIdx, PosEdges)
        idxArray[posMask] = posIdx + n_negative_classes - 1
    return idxArray


def class_index_abs_log(v_array, v_log_edges):
    """
    input: array of velocities, edges for log of velocity
    output: index of velocity class. for each absolute value of velocity there are
    two classes the first one corresponding to positive direction
    """
    abs_v = np.abs(v_array)
    v_index = np.digitize(np.log(abs_v), v_log_edges)
    fix_out_of_bound(v_index, v_log_edges)
    v_index -= 1
    v_sign = np.sign(v_array)
    return 2 * v_index + (v_sign < 0)


def fix_out_of_bound(idx_array, bins):
    """
    idx_array is the return values of np.digitize(input, bins)
    set out of left bound to first bin
    """
    len_bins = len(bins)
    idx_array[idx_array == 0] += 1
    idx_array[idx_array == len_bins] -= 1


def fix_bisect_left_indices(idx_array, bisect_array):
    """
    :param idx_arrray: indices returned by bisect_left
    :param bisect_array: initially bisected array
    :return: idx_array with large out of bound indices reduced by one
    """
    array_len = len(bisect_array)
    idx_array[idx_array == array_len] -= 1

def make_1d_equiprob_velocity_bins(big_v_array, n_class, n_neg_class):
    sort_idx = np.argsort(big_v_array)
    sorted_v = big_v_array[sort_idx]
    # log of dx
    # find the positive entries
    pos_idx = sorted_v > 0.0
    pos_sorted_v = sorted_v[pos_idx]
    neg_sorted_v = sorted_v[~pos_idx]
    log_pos_sorted_v = np.log(pos_sorted_v)
    log_neg_sorted_v = np.log(-neg_sorted_v)
    n_pos_class = n_class - n_neg_class
    eps = 2e-14
    # Bin edges for log of positive values
    vLogMin = np.amin(log_pos_sorted_v)
    vLogMax = np.amax(log_pos_sorted_v)
    percentile_array = np.linspace(0, 100, n_pos_class + 1)
    v_pos_log_edges = np.percentile(log_pos_sorted_v, percentile_array)
    v_pos_log_edges[0] = vLogMin - eps
    v_pos_log_edges[-1] = vLogMax + eps
    # Bin edges for minus log of negative values --> equal prob
    percentile_array = np.linspace(0, 100, n_neg_class + 1)
    v_neg_log_edges = np.percentile(log_neg_sorted_v, percentile_array)
    v_neg_log_edges[0] -= eps
    v_neg_log_edges[-1] += eps
    class_velocity = np.zeros(n_class)
    for i in range(n_neg_class):
        log_value = 0.5 * (v_neg_log_edges[i] + v_neg_log_edges[i + 1])
        class_velocity[i] = -1.0 * np.exp(log_value)
    for i in range(n_pos_class):
        log_value = 0.5 * (v_pos_log_edges[i] + v_pos_log_edges[i + 1])
        class_velocity[n_neg_class + i] = np.exp(log_value)
    return v_neg_log_edges, v_pos_log_edges, class_velocity


def make_1d_equiprob_abs_vel_bins(big_v_array, n_class):
    """
    function to make equal probability absolute velocity bins
    """
    abs_v = np.abs(big_v_array)
    sort_idx = np.argsort(abs_v)
    log_sorted_abs_v = np.log(abs_v[sort_idx])
    # Bin edges
    percentile_array = np.linspace(0, 100, n_class + 1)
    v_log_edges = np.percentile(log_sorted_abs_v, percentile_array)
    # #add a special bin for the very rare small velocities
    # nbins = 150
    # h,bins = np.histogram(np.log(np.abs(big_v_array)), bins = nbins)
    # center_vals = 0.5*np.diff(bins) + bins[:-1]
    # cdf_v = np.array(get_cdf(h))
    # idx_cut = np.where(cdf_v > 0.001)[0][0]
    # v_log_edges = np.insert(v_log_edges, 1, center_vals[idx_cut])
    return v_log_edges


def make_1d_abs_vel_bins(big_v_array, n_class, n_slow_classes=2):
    """
    function to make almost equal probability absolute velocity bins,
    there are more bins added for very slow velocity values to resolve that
    region better.
    """
    abs_v = np.abs(big_v_array)
    sort_idx = np.argsort(abs_v)
    log_sorted_abs_v = np.log(abs_v[sort_idx])
    # Bin edges
    percentile_array = np.linspace(0, 100, n_class + 1 - n_slow_classes)
    v_log_edges = np.percentile(log_sorted_abs_v, percentile_array)
    # add a special bin for the very rare small velocities
    nbins = 150
    h, bins = np.histogram(np.log(np.abs(big_v_array)), bins=nbins)
    center_vals = 0.5 * np.diff(bins) + bins[:-1]
    cdf_v = np.array(get_cdf(h))
    idx_cut = np.where(cdf_v > 0.001)[0][0]
    temp_edges = np.linspace(v_log_edges[0], center_vals[idx_cut], n_slow_classes + 1)
    v_log_edges = np.insert(v_log_edges, 1, temp_edges[1:])
    assert (len(v_log_edges) - 1 == n_class)
    return v_log_edges


def abs_vel_log_bins_low_high(big_v_array, n_class, n_low = 5, max_allowed = 0.5, init_percentile=None):
    """
    make bins for absolute value of log of velocity.
    First we make equal probability bins, then resolve more for very slow and very fast velocities.
    :param big_v_array: input velocity
    :param n_class: initial attempt for number of classes
    :param n_low: the number of subclasses in the slow tail
    :param max_allowed: maximum bin width allowed for fast velocities
    :return: edges for log of velocity
    """
    if not init_percentile:
        init_percentile = 0
    abs_v = np.abs(big_v_array)
    sort_idx = np.argsort(abs_v)
    log_sorted_abs_v = np.log(abs_v[sort_idx])
    # Bin edges
    percentile_array = np.linspace(init_percentile, 100, n_class + 1 - n_low)
    v_log_edges = np.percentile(log_sorted_abs_v, percentile_array)
    #look at the very rare small velocities and add more bins there
    nbins = 300
    h, bins = np.histogram(np.log(np.abs(big_v_array)), bins=nbins)
    center_vals = 0.5 * np.diff(bins) + bins[:-1]
    cdf_v = np.array(get_cdf(h))
    idx_cut_low = np.where(cdf_v > 0.005)[0][0]
    #find the first bin edge below this value
    idx_insert_low = bs.bisect_left(v_log_edges, center_vals[idx_cut_low])
    temp_edges = np.linspace(v_log_edges[idx_insert_low-1], center_vals[idx_cut_low], n_low+1)
    v_log_edges = np.insert(v_log_edges, idx_insert_low, temp_edges[1:])
    if max_allowed:
        #check the last 12 bins and split the bin if it is larger than maximum allowed size
        bin_width = np.diff(v_log_edges)
        for i in np.arange(-12,-1):
            if bin_width[i+1] > max_allowed:
                n_split = np.ceil(bin_width[i+1]/max_allowed)
                print 'splitting bin ', str(i+1), 'to ', str(n_split)
                temp = np.linspace(v_log_edges[i], v_log_edges[i+1], n_split+1)
                v_log_edges = np.insert(v_log_edges, i+1, temp[1:-1])
    print 'number of output bins: ', len(v_log_edges) -1
    return v_log_edges


def make_theta_bins_linear(n_theta_classes):
    """
    function to define theta bin edges, output edges span between -pi and pi
    :param n_theta_classes:
    :return: theta_edges
    """
    d_theta = 2*np.pi/n_theta_classes
    theta_edges = np.arange(0,n_theta_classes+1)*d_theta - np.pi
    return theta_edges

# TODO: experiment with different resolutions for y, most simple case for now
def make_y_bins_linear(big_y_array, n_y_classes):
    """
    make linear bins for distance from the origin
    :param big_y_array: sample observed values for y
    :param n_y_classes: number of y_classes
    :return: y_bin_edges
    """
    ymax = np.max(np.abs(big_y_array))
    y_edges = np.linspace(-ymax, ymax, n_y_classes+1)
    return y_edges

def make_input_for_binning_no_freq(input_folder, n_realizations, prefix='real', verbose=True):
    if verbose:
        print "making long array for generating bins..."
    total_length = 0
    # each realization has 1000 particles
    pointer_list = []
    initial_v0 = []
    initial_v1 = []
    big_v_array = np.array([], dtype=np.float)
    for j in range(n_realizations):
        print "reading realization nr ", j
        file_name = prefix + "_" + str(j) + ".pkl"
        input_file = os.path.join(input_folder, file_name)
        with open(input_file, 'rb') as input:
            dataHolder = pickle.load(input)
        dx = np.diff(dataHolder.x_array)
        dt = np.diff(dataHolder.t_array)
        lastIdx = dataHolder.last_idx_array
        vMatrix = np.divide(dx, dt)
        assert (np.all(vMatrix[:, 0] > 0.0))
        m = dx.shape[0]
        for i in range(m):
            cutOff = lastIdx[i]
            small_v = vMatrix[i, 0:cutOff - 1]
            current_length = cutOff - 1
            if current_length > 1:
                total_length += current_length
                big_v_array = np.hstack((big_v_array, small_v))
                pointer_list.append(total_length)
                # save the first velocity for initialization
                initial_v0.append(small_v[0])
                initial_v1.append(small_v[1])
    initial_v0 = np.array(initial_v0)
    initial_v1 = np.array(initial_v1)
    return big_v_array, pointer_list, initial_v0, initial_v1


def make_input_for_binning_with_freq(input_folder, n_realizations, time_step, prefix='real', verbose=True):
    if verbose:
        print "making long array for generating bins..."
    total_length = 0
    # each realization has 1000 particles
    pointer_list = []
    initial_v0 = []
    initial_f0 = []
    big_v_array = np.array([], dtype=np.float)
    big_freq_array = np.array([], dtype=np.float)
    for j in range(n_realizations):
        print "reading realization nr ", j
        case_name = prefix + "_" + str(j) + ".pkl"
        input_file = os.path.join(input_folder, case_name)
        with open(input_file, 'rb') as input:
            dataHolder = pickle.load(input)
        dx = np.diff(dataHolder.x_array)
        dt = np.diff(dataHolder.t_array) + 1e-16
        lastIdx = dataHolder.last_idx_array
        vMatrix = np.divide(dx, dt)
        m = dx.shape[0]
        for i in range(m):
            # get the time process for each velocity
            cutOff = lastIdx[i]
            dxTime, freq = get_time_dx_array_with_frequency(dt[i, :cutOff], vMatrix[i, :cutOff], time_step)
            current_length = len(dxTime)
            if current_length > 1:
                total_length += current_length
                big_v_array = np.hstack((big_v_array, dxTime))
                big_freq_array = np.hstack((big_freq_array, freq))
                pointer_list.append(total_length)
                # save the first velocity for initialization
                initial_v0.append(dxTime[0] / time_step)
                initial_f0.append(freq[0])
    assert (len(big_v_array) == len(big_freq_array))
    initial_v0 = np.array(initial_v0)
    initial_f0 = np.array(initial_f0)
    # divide by time to get velocity
    big_v_array /= time_step
    return big_v_array, big_freq_array, pointer_list, initial_v0, initial_f0

def make_input_for_binning_v_theta_freq(input_folder, n_realizations, time_step, prefix='real', verbose=True, print_every=20):
    """
    :param input_folder: folder containing the input realizations
    :param n_realizations: number of realizations to consider
    :param time_step: time step size
    :param prefix: prefix for input files
    :param verbose: whether to write output messages or not
    :return big_v_array:
    :return big_freq_array:
    :return big_theta_array:
    :return pointer_list:
    :return initial_v0:
    :return initial f_0:
    :return initial_theta0:
    """
    if verbose:
        print "making long array for generating v, theta, frequency bins..."
    total_length = 0
    # each realization has 1000 particles
    pointer_list = []
    initial_v = []
    initial_f = []
    initial_theta = []
    big_v_list, big_theta_list, big_freq_list = [[] for i in range(3)]
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
        for i in range(m):
            x_start = dataHolder.x_array[i,0]
            y_start = dataHolder.y_array[i,0]
            # get the time process for each velocity
            cutOff = lastIdx[i]
            dxTime, dyTime, freq = get_time_dx_dy_array_with_freq(dt[i, :cutOff], vxMatrix[i, :cutOff], vyMatrix[i, :cutOff], x_start, y_start, time_step)
            if len(dxTime) < 1:
                continue
            dxTime, dyTime, freq = remove_duplicate_xy(dxTime, dyTime, freq)
            current_length = len(dxTime)
            if current_length > 1:
                total_length += current_length
                current_v = np.sqrt(np.power(dxTime,2) + np.power(dyTime,2))
                current_theta = np.arctan2(dyTime, dxTime)
                big_v_list.append(current_v)
                big_theta_list.append(current_theta)
                big_freq_list.append(freq)
                pointer_list.append(total_length)
                # save the first velocity for initialization
                initial_v.append(current_v[0] / time_step)
                initial_theta.append(current_theta[0])
                initial_f.append(freq[0])
    # flatten the big lists
    chain = itertools.chain(*big_v_list)
    big_v_array = np.array(list(chain), dtype=np.float)
    chain = itertools.chain(*big_theta_list)
    big_theta_array = np.array(list(chain), dtype=np.float)
    chain = itertools.chain(*big_freq_list)
    big_freq_array = np.array(list(chain), dtype=np.float)
    assert (len(big_v_array) == len(big_freq_array))
    initial_v = np.array(initial_v)
    initial_f = np.array(initial_f)
    initial_theta = np.array(initial_theta)
    # divide by time to get velocit
    big_v_array /= time_step
    return big_v_array, big_theta_array, big_freq_array, pointer_list, initial_v, initial_f, initial_theta

def make_input_for_binning_v_theta_freq_with_filter(input_folder, n_realizations, time_step, filter_time,
                                                    prefix='real', verbose=True):
    """
    :param input_folder: folder containing the input realizations
    :param n_realizations: number of realizations to consider
    :param time_step: time step size
    :param filter_time: time to filter on
    :param prefix: prefix for input files
    :param verbose: whether to write output messages or not
    :return big_v_array:
    :return big_freq_array:
    :return big_theta_array:
    :return pointer_list:
    :return initial_v0:
    :return initial f_0:
    :return initial_theta0:
    """
    if verbose:
        print "making long array for generating v, theta, frequency bins..."
    total_length = 0
    # each realization has 1000 particles
    pointer_list = []
    initial_v = []
    initial_f = []
    initial_theta = []
    big_v_array = np.array([], dtype=np.float)
    big_theta_array = np.array([], dtype=np.float)
    big_freq_array = np.array([], dtype=np.float)
    for j in range(n_realizations):
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
        for i in range(m):
            x_start = dataHolder.x_array[i,0]
            y_start = dataHolder.y_array[i,0]
            # get the time process for each velocity
            cutOff = min(lastIdx[i], np.argmin(dataHolder.t_array < filter_time))
            dxTime, dyTime, freq = get_time_dx_dy_array_with_freq(dt[i, :cutOff], vxMatrix[i, :cutOff], vyMatrix[i, :cutOff], x_start, y_start, time_step)
            if len(dxTime) < 1:
                continue
            dxTime, dyTime, freq = remove_duplicate_xy(dxTime, dyTime, freq)
            current_length = len(dxTime)
            if current_length > 1:
                total_length += current_length
                current_v = np.sqrt(np.power(dxTime,2) + np.power(dyTime,2))
                current_theta = np.arctan2(dyTime, dxTime)
                big_v_array = np.hstack((big_v_array, current_v))
                big_theta_array = np.hstack((big_theta_array, current_theta))
                big_freq_array = np.hstack((big_freq_array, freq))
                pointer_list.append(total_length)
                # save the first velocity for initialization
                initial_v.append(current_v[0] / time_step)
                initial_theta.append(current_theta[0])
                initial_f.append(freq[0])
    assert (len(big_v_array) == len(big_freq_array))
    initial_v = np.array(initial_v)
    initial_f = np.array(initial_f)
    initial_theta = np.array(initial_theta)
    # divide by time to get velocit
    big_v_array /= time_step
    return big_v_array, big_theta_array, big_freq_array, pointer_list, initial_v, initial_f, initial_theta

def binning_input_v_theta_freq_y(input_folder, n_realizations, time_step, prefix='real', verbose=True):
    """
    generate sample processes for v, theta, freq, y to be used for creating classes
    :param input_folder: folder containing the input realizations
    :param n_realizations: number of realizations to consider
    :param time_step: time step size
    :param prefix: prefix for input files
    :param verbose: whether to write output messages or not
    :return big_v_array:
    :return big_freq_array:
    :return big_theta_array:
    :return pointer_list:
    :return initial_v0:
    :return initial f_0:
    :return initial_theta0:
    """
    if verbose:
        print "making long array for generating v, theta, frequency bins..."
    total_length = 0
    #
    pointer_list = []
    initial_v = []
    initial_f = []
    initial_theta = []
    big_v_array = np.array([], dtype=np.float)
    big_theta_array = np.array([], dtype=np.float)
    big_freq_array = np.array([], dtype=np.float)
    big_y_array = np.array([], dtype=np.float)
    for j in range(n_realizations):
        if verbose:
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
        for i in range(m):
            x_start = dataHolder.x_array[i,0]
            y_start = dataHolder.y_array[i,0]
            # get the time process for each velocity (averaging/integrating arrays in time)
            cutOff = lastIdx[i]
            dxTime, dyTime, freq = get_time_dx_dy_array_with_freq(dt[i, :cutOff], vxMatrix[i, :cutOff], vyMatrix[i, :cutOff], x_start, y_start, time_step)
            if len(dxTime) < 1:
                continue
            dxTime, dyTime, freq = remove_duplicate_xy(dxTime, dyTime, freq)
            # find y
            y_time = np.hstack((0.0, np.cumsum(dyTime)))
            current_length = len(dxTime)
            if current_length > 1:
                total_length += current_length
                current_v = np.sqrt(np.power(dxTime,2) + np.power(dyTime,2))
                current_theta = np.arctan2(dyTime, dxTime)
                big_v_array = np.hstack((big_v_array, current_v))
                big_theta_array = np.hstack((big_theta_array, current_theta))
                big_freq_array = np.hstack((big_freq_array, freq))
                big_y_array = np.hstack((big_y_array, y_time))
                pointer_list.append(total_length)
                # save the first velocity for initialization
                initial_v.append(current_v[0] / time_step)
                initial_theta.append(current_theta[0])
                initial_f.append(freq[0])
    assert (len(big_v_array) == len(big_freq_array))
    initial_v = np.array(initial_v)
    initial_f = np.array(initial_f)
    initial_theta = np.array(initial_theta)
    # divide by time to get velocit
    big_v_array /= time_step
    return big_v_array, big_theta_array, big_y_array, big_freq_array, \
           pointer_list, initial_v, initial_f, initial_theta

def flatten(ll):
    chain = itertools.chain(*ll)
    return np.array(list(chain), dtype=np.float)

def make_input_for_binning_v_theta(input_folder, n_realizations, prefix='real', verbose=True, print_every=20):
    """
    :param input_folder: folder containing the input realizations
    :param n_realizations: number of realizations to consider
    :param prefix: prefix for input files
    :param verbose: whether to write output messages or not
    :return big_v_array:
    :return big_freq_array:
    :return big_theta_array:
    :return pointer_list:
    :return initial_v0:
    :return initial f_0:
    :return initial_theta0:
    """
    if verbose:
        print "making long array for generating v, theta, frequency bins..."
    total_length = 0
    pointer_list, initial_v, initial_theta, big_v_list, big_theta_list = [[] for i in range(5)]
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
        vx_matrix = np.divide(dx, dt)
        vy_matrix = np.divide(dy, dt)
        m = dx.shape[0]
        for i in range(m):
            cut_off = lastIdx[i]
            vx, vy = vx_matrix[i, :cut_off], vy_matrix[i, :cut_off]
            current_length = len(vx)
            if current_length > 1:
                total_length += current_length
                current_v = np.sqrt(np.power(vx,2) + np.power(vy,2))
                current_theta = np.arctan2(vy, vx)
                big_v_list.append(current_v)
                big_theta_list.append(current_theta)
                pointer_list.append(total_length)
                # save the first velocity for initialization
                initial_v.append(current_v[0])
                initial_theta.append(current_theta[0])
    # flatten the big lists
    big_v_array, big_theta_array = [flatten(ll) for ll in [big_v_list, big_theta_list]]
    initial_v = np.array(initial_v)
    initial_theta = np.array(initial_theta)
    return big_v_array, big_theta_array, pointer_list, initial_v, initial_theta