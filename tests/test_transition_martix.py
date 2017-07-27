import numpy as np
from copy import copy
from py_dp.dispersion.transition_matrix_fcns import class_index_log_log
from py_dp.dispersion.transition_matrix_fcns import count_matrix_with_lag

def test_class_idx_log_log():
    #testing class_index
    v_test_positive = np.exp([-6.5, -5.0, -3.5, -1.0,  0.0, 0.5])
    v_test_negative = -1.0*np.exp([-2.5, 0.5])
    pos_log_edges = np.array([-7.0, -5.0, -3.0, 0.0, 1.0, 2.0, 2.5])
    neg_log_edges = np.array([-3, -2, 1.0])
    #test1 - only positive values
    v_values = copy(v_test_positive)
    expected_output = 2.0 + np.array([1,2,2,3,4,4]) - 1.0
    test_output = class_index_log_log(v_values, neg_log_edges, pos_log_edges)
    print "output: ", test_output
    assert(np.all(expected_output == test_output))
    #test2 - only negative values
    v_values = copy(v_test_negative)
    expected_output = np.array([0.0, 1.0])
    test_output = class_index_log_log(v_values, neg_log_edges, pos_log_edges)
    print "output: ", test_output
    assert(np.all(expected_output == test_output))
    #test3 - mixed values
    pos_idx = [0,1,3,4,6,7]
    neg_idx = [2,5]
    v_values = np.zeros(8)
    v_values[pos_idx] = v_test_positive
    v_values[neg_idx] = v_test_negative
    expected_output = np.array([2.0, 3.0, 0.0, 3.0, 4.0, 1.0, 5.0, 5.0])
    test_output = class_index_log_log(v_values, neg_log_edges, pos_log_edges)
    print "output: ", test_output
    assert(np.all(expected_output == test_output))

def test_transition_count_with_frequency():
    #testing count_matrix
    #test3 - mixed values
    v_test_positive = np.exp([-6.5, -5.0, -3.5, -1.0,  0.0, 0.5])
    v_test_negative = -1.0*np.exp([-2.5, 0.5])
    pos_log_edges = np.array([-7.0, -5.0, -3.0, 0.0, 1.0, 2.0, 2.5])
    neg_log_edges = np.array([-3, -2, 1.0])
    n_class = 8.0
    test_pointer = [8]
    pos_idx = [0,1,3,4,6,7]
    neg_idx = [2,5]
    v_values = np.zeros(8)
    v_values[pos_idx] = v_test_positive
    v_values[neg_idx] = v_test_negative
    expected_output = np.array([2.0, 3.0, 0.0, 3.0, 4.0, 1.0, 5.0, 5.0])
    v_frequency =     np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    test_output = class_index_log_log(v_values, neg_log_edges, pos_log_edges)
    assert(np.all(expected_output == test_output))
    lag = 1
    count_matrix = count_matrix_with_lag(lag, n_class, v_values, v_frequency,
                                         test_pointer, neg_log_edges, pos_log_edges)
    expected_matrix = np.zeros((n_class, n_class))
    #transition from zero
    expected_matrix[0,0] = 2.0 
    expected_matrix[3,0] = 1.0 
    #transition from one
    expected_matrix[1,1] = 5.0 
    expected_matrix[5,1] = 1.0 
    #transition from two
    expected_matrix[2,2] = 0.0 
    expected_matrix[3,2] = 1.0
    #transition from three
    expected_matrix[3,3] = 4.0 
    expected_matrix[0,3] = 1.0
    expected_matrix[4,3] = 1.0
    #transition from four
    expected_matrix[4,4] = 4.0 
    expected_matrix[1,4] = 1.0
    #transition from five
    expected_matrix[5,5] = 14.0 
    assert np.all(expected_matrix==count_matrix)
    ##try for another lag
    lag = 3
    count_matrix = count_matrix_with_lag(lag, n_class, v_values, v_frequency,
                                         test_pointer, neg_log_edges, pos_log_edges)
    expected_matrix = np.zeros((n_class, n_class))
    #transition from zero
    expected_matrix[3,0] = 3.0 
    #transition from one
    expected_matrix[1,1] = 3.0 
    expected_matrix[5,1] = 3.0 
    #transition from two
    expected_matrix[2,2] = 0.0 
    expected_matrix[0,2] = 1.0
    #transition from three
    expected_matrix[3,3] = 1.0 
    expected_matrix[0,3] = 2.0
    expected_matrix[4,3] = 3.0
    #transition from four
    expected_matrix[4,4] = 2.0 
    expected_matrix[1,4] = 3.0
    #transition from five
    expected_matrix[5,5] = 12.0 
    #print count_matrix
    assert np.all(expected_matrix==count_matrix)