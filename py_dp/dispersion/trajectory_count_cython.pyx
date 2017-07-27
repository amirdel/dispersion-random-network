import numpy as np
cimport numpy as np
DTYPE = np.int
ctypedef np.int_t DTYPE_t
DTYPE2 = np.float
ctypedef np.float_t DTYPE2_t
from py_dp.dispersion.binning import class_index_log_log
from py_dp.dispersion.binning import class_index_abs_log
from py_dp.dispersion.binning import fix_out_of_bound
from py_dp.dispersion.second_order_markov import find_2d_bin
from py_dp.dispersion.binning import get_cdf
import random as random
import bisect as bs
import copy

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
def count_with_lag_one_trajectory_cython(np.ndarray[double, ndim=2] transition_count_matrix, 
                                         np.int lag, np.ndarray[double] v_array, 
                                         np.ndarray[double] v_neg_log_edges, 
                                         np.ndarray[double] v_pos_log_edges):
    cdef int j, nextBin, currentBin
    cdef np.ndarray[DTYPE_t] inds_temp
    inds_temp = class_index_log_log(v_array,v_neg_log_edges,v_pos_log_edges)
    for j in range(len(v_array) - lag):
        nextBin = inds_temp[j+lag]
        currentBin = inds_temp[j]
        transition_count_matrix[nextBin,currentBin] += 1

def count_with_lag_one_trajectory_kang_cython(np.ndarray[double, ndim=2] transition_count_matrix, 
                                              np.int lag, np.ndarray[double] v_array, 
                                              np.ndarray[double] v_log_edges):
    cdef int j, nextBin, currentBin
    cdef np.ndarray[DTYPE_t] inds_temp
    inds_temp = class_index_abs_log(v_array,v_log_edges)
    for j in range(len(v_array) - lag):
        nextBin = inds_temp[j+lag]
        currentBin = inds_temp[j]
        transition_count_matrix[nextBin,currentBin] += 1

@cython.boundscheck(False) # turn off bounds-checking for entire function
def count_with_lag_one_trajectory_aggr_cython(np.ndarray[double, ndim=2] transition_count_matrix, 
                                              np.int lag, np.ndarray[double] v_array, 
                                              np.ndarray[double] v_log_edges):
    cdef int j, nextBin, currentBin, len_bins_array
    cdef np.ndarray[DTYPE_t] inds_temp
    len_bins_array = len(v_log_edges)
    inds_temp = np.digitize(np.log(np.abs(v_array)), v_log_edges)
    fix_out_of_bound(inds_temp, v_log_edges)
    inds_temp -= 1
    for j in range(len(v_array) - lag):
        nextBin = inds_temp[j+lag]
        currentBin = inds_temp[j]
        transition_count_matrix[nextBin,currentBin] += 1

def fill_2nd_order_one_trajectory_cython(np.int lag, np.ndarray[double] v_array, 
                                         np.ndarray[double] v_log_edges, i_list, 
                                         j_list, ij_list, val_list, np.int n_class):
    cdef int j, nextBin, currentBin
    cdef np.ndarray[DTYPE_t] inds_temp
    inds_temp = class_index_abs_log(v_array,v_log_edges)
    for j in range(len(v_array) - lag):
        current_bin = find_2d_bin(inds_temp[j-1], inds_temp[j], n_class)
        next_bin = find_2d_bin(inds_temp[j+lag-1], inds_temp[j+lag], n_class)
        #transition_count_matrix[next_bin, current_bin] += 1
        if not (next_bin, current_bin) in ij_list:
            i_list.append(next_bin)
            j_list.append(current_bin)
            val_list.append(1)
        else:
            mask = ((i_list == next_bin) and (j_list == current_bin))
            val_list[mask] += 1

@cython.boundscheck(False) # turn off bounds-checking for entire function
def fill_one_trajectory_sparse_cython(np.int lag, np.ndarray[DTYPE_t] class_array,
                                      i_list, j_list, ij_list, val_list):
    cdef int j, nextBin, currentBin
    for j in range(len(class_array) - lag):
        current_bin = class_array[j]
        next_bin = class_array[j+lag]
        if not (next_bin, current_bin) in ij_list:
            i_list.append(next_bin)
            j_list.append(current_bin)
            val_list.append(1)
        else:
            mask = ((i_list == next_bin) and (j_list == current_bin))
            val_list[mask] += 1

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
def fill_one_trajectory_sparse_with_freq_cython(int lag, np.ndarray[DTYPE_t] class_array, np.ndarray[DTYPE_t] freq_array,
                                                i_list, j_list, ij_list, val_list):
    """
    Count matrix in time using kang bins
    """
    if not len(class_array):
        return
    cdef int idx_cum1, idx_cum2, last_idx, n_repeating_pair, added_val
    cdef int next_bin, current_bin
    idx_cum1 = 1
    idx_cum2 = 1 + lag
    cdef np.ndarray[DTYPE_t] cumsum_temp
    cumsum_temp = np.cumsum(freq_array)
    cdef int ll = len(cumsum_temp)
    last_index = cumsum_temp[ll-1]
    while idx_cum2 < last_index:
        [idx1, idx2] = np.digitize([idx_cum1, idx_cum2], cumsum_temp, right=True)
        if idx1 != idx2:
            #in this case they are not in the same cumsum_bin hence no repetition
            #current bin is idx1
            next_bin = class_array[idx2]
            current_bin = class_array[idx1]
            added_val = 1
        else:
            #they are in a bin with repetition might happen
            #number of repeating pair
            added_val = (cumsum_temp[idx1] - idx_cum2) + 1
            next_bin = class_array[idx1]
            current_bin = next_bin

        if not (next_bin, current_bin) in ij_list:
            i_list.append(next_bin)
            j_list.append(current_bin)
            val_list.append(added_val)
        else:
            mask = ((i_list == next_bin) and (j_list == current_bin))
            val_list[mask] += added_val
        idx_cum1 += added_val
        idx_cum2 += added_val

def get_cdf_cython(np.ndarray[DTYPE2_t] input_array):
    # sum of array should be one
    input_array = input_array / float(np.sum(input_array))
    cdef np.ndarray[DTYPE2_t] cdf = np.zeros(len(input_array), dtype=DTYPE2)
    cdf[0] = input_array[0]
    cdef int i
    for i in xrange(1, len(input_array)):
        cdf[i] = cdf[i-1] + input_array[i]
    return cdf

def choose_next_class_vector_cython(np.ndarray[DTYPE_t] indptr, np.ndarray[DTYPE_t] indices, np.ndarray[DTYPE2_t] data,
                                    np.ndarray[DTYPE_t] current_class):
    """
    function to draw the next class based on a csc transition matrix
    """
    cdef np.ndarray[DTYPE_t] next_class, rows
    cdef np.ndarray[DTYPE2_t] cdf, values
    next_class = np.zeros(len(current_class), dtype=DTYPE)
    cdef int i, current_idx, start, end
    cdef np.ndarray[DTYPE2_t] rand_array = np.random.rand(len(current_class))
    for i in range(len(current_class)):
        current_idx = current_class[i]
        start = indptr[current_idx]
        end = indptr[current_idx+1]
        rows = indices[start:end]
        values = data[start:end]
        if len(values) == 0:
            next_class[i] = -12
        else:
            cdf = get_cdf_cython(values)
            next_class[i] = rows[bs.bisect_left(cdf, rand_array[i])]
    return next_class