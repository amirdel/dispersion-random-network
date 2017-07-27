import numpy as np
from py_dp.dispersion.transition_matrix_fcns import class_index_log_log
from py_dp.dispersion.second_order_markov import  count_2nd_markov_sparse, find_1d_bins, find_2d_bin

def test_transition_second_order():
    # testing count_matrix
    # test3 - mixed values
    v_test_positive = np.exp([0.5,0.5,0.5,0.5])
    v_test_negative = -1.0*np.exp([0.5,0.5,0.5,0.5])
    pos_log_edges = np.array([0.1, 0.8])
    neg_log_edges = np.array([0.1, 0.8])
    n_class = 2.0
    test_pointer = [8]
    neg_idx = [0,2,4,6]
    pos_idx = [1,3,5,7]
    v_values = np.zeros(8)
    v_values[pos_idx] = v_test_positive
    v_values[neg_idx] = v_test_negative
    expected_output = np.array([0,1,0,1,0,1,0,1])
    test_output = class_index_log_log(v_values, neg_log_edges, pos_log_edges)
    print test_output
    assert(np.all(expected_output == test_output))
    lag = 1
    count_matrix = count_2nd_markov_sparse(lag, n_class, v_values, test_pointer, neg_log_edges, pos_log_edges)
    expected_matrix = np.zeros((4,4))
    #transition from zero
    expected_matrix[1,2] = 3.0
    expected_matrix[2,1] = 3.0
    assert np.all(expected_matrix==count_matrix.todense())


    v_test_positive = np.exp([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    v_test_negative = -1.0*np.exp([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    pos_log_edges = np.array([0.1, 0.8])
    neg_log_edges = np.array([0.1, 0.8])
    n_class = 2.0
    test_pointer = [12]
    neg_idx = [0,1,3,7,8,11]
    pos_idx = [2,4,5,6,9,10]
    v_values = np.zeros(12)
    v_values[pos_idx] = v_test_positive
    v_values[neg_idx] = v_test_negative
    expected_output = np.array([0,0,1,0,1,1,1,0,0,1,1,0])
    test_output = class_index_log_log(v_values, neg_log_edges, pos_log_edges)
    print test_output
    assert(np.all(expected_output == test_output))
    lag = 1
    count_matrix = count_2nd_markov_sparse(lag, n_class, v_values, test_pointer, neg_log_edges, pos_log_edges)
    expected_matrix = np.zeros((4,4))
    #transition from zero
    expected_matrix[1,0] = 2.0
    expected_matrix[2,1] = 1.0
    expected_matrix[1,2] = 1.0
    expected_matrix[3,1] = 2.0
    expected_matrix[3,3] = 1.0
    expected_matrix[2,3] = 2.0
    expected_matrix[0,2] = 1.0
    print count_matrix.todense()
    assert np.all(expected_matrix==count_matrix.todense())
