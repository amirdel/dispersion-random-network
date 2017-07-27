import numpy as np
from py_dp.dispersion.binning import class_index_log_log
from scipy.sparse import csc_matrix

def find_2d_bin(i,j,n_class):
    """
    i is the first velocity class for v_n, j is the velocity class for v_n+1
    n_class is the number of classes for the order one Markov chain
    the function returns the class for the pair (v_n, v_n+1)
    """
    assert(i<n_class and j<n_class)
    return (i)*n_class + j

def find_1d_bins(bin_2d ,n_class):
    """
    given the pair for (v_n, v_n+1) find the velocity class for v_n, v_n+1
    """
    j = np.mod(bin_2d, n_class)
    i = (bin_2d - j)/n_class
    return i,j

def count_2nd_markov(lag, n_class, v_array, pointer_list, v_neg_log_edges, v_pos_log_edges):
    n_class_square = n_class*n_class
    transition_count_matrix = np.zeros((n_class_square, n_class_square))
    start_idx = 0
    for i in pointer_list:
        v_temp = v_array[start_idx:i]
        start_idx = i
        inds_temp = class_index_log_log(v_temp,v_neg_log_edges,v_pos_log_edges)
        for j in range(1,len(v_temp) - lag):
            current_bin = find_2d_bin(inds_temp[j-1], inds_temp[j], n_class)
            next_bin = find_2d_bin(inds_temp[j+lag-1], inds_temp[j+lag], n_class)
            transition_count_matrix[next_bin, current_bin] += 1
    return transition_count_matrix

def count_2nd_markov_sparse(lag, n_class, v_array, pointer_list, v_neg_log_edges, v_pos_log_edges):
    n_class_square = n_class*n_class
    i_list = []
    j_list = []
    ij_list = set([])
    val_list = []
    start_idx = 0
    for i in pointer_list:
        v_temp = v_array[start_idx:i]
        start_idx = i
        inds_temp = class_index_log_log(v_temp,v_neg_log_edges,v_pos_log_edges)
        for j in range(1,len(v_temp) - lag):
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
    return csc_matrix((val_list, (i_list, j_list)), shape = (n_class_square, n_class_square))