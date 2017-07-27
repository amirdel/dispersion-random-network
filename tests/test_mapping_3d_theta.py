from py_dp.dispersion.binning import make_input_for_binning_with_freq, make_1d_abs_vel_bins, class_index_abs_log, make_theta_bins_linear
from py_dp.dispersion.convert_to_time_process_with_freq import remove_duplicate, remove_duplicate_xy
import numpy as np
from py_dp.dispersion.mapping import mapping_v_theta_repeat
from py_dp.dispersion.transition_matrix_fcns import fix_out_of_bound
import os
from py_dp.dispersion.binning import make_input_for_binning_v_theta_freq, abs_vel_log_bins_low_high

def test_mapping_with_theta():
    main_folder = os.path.dirname(os.path.dirname(__file__))
    input_folder = os.path.join(main_folder, 'test_related_files','particle_tracking_results')
    dt = 50.0
    n_realz = 1
    big_v_array, big_freq_array, pointer_list, initial_v0, initial_v1 = make_input_for_binning_with_freq(input_folder,
                                                                                                         n_realz, dt)
    new_v, new_f = remove_duplicate(big_v_array, big_freq_array)
    #make random angles
    new_theta = np.random.randn(len(new_v))*np.pi
    n_abs_log_class = 8
    n_theta_classes = 8
    abs_log_v_edges = make_1d_abs_vel_bins(new_v, n_abs_log_class, n_slow_classes = 1)
    theta_bin_edges = make_theta_bins_linear(n_theta_classes)
    v_class_number = class_index_abs_log(new_v, abs_log_v_edges)

    mapping = mapping_v_theta_repeat(abs_log_v_edges, theta_bin_edges, new_v, new_theta, new_f)
    # find 1d v and theta bins, calculate the 2d bins, calculate the 1d bins from 2d bins
    v_idx = np.digitize(np.log(np.abs(new_v)), abs_log_v_edges)
    fix_out_of_bound(v_idx, abs_log_v_edges)
    v_idx -= 1
    theta_idx = np.digitize(new_theta, theta_bin_edges)
    fix_out_of_bound(theta_idx, theta_bin_edges)
    theta_idx -= 1
    idx_2d = mapping.class_index_2d_vtheta(new_v, new_theta)
    v_idx2, theta_idx2 = mapping.class_index_1d_v_theta_from_2d(idx_2d)
    assert (np.all(v_idx2 == v_idx))
    assert (np.all(theta_idx2 == theta_idx))

    # test inverse mapping
    class_3d_array = range(mapping.n_3d_classes)
    for class_3d_test in class_3d_array:
        abs_v, sgn_v, freq = mapping.find_v_theta_freq(class_3d_test)
    # test draw velocity
    v_log_edges = mapping.v_log_edges
    # for i in range(len(v_log_edges)-1):
    for i in range(mapping.n_abs_v_classes):
        class_2d = i
        v1 = mapping.draw_from_class_velocity(class_2d)
        assert (np.log(v1) > v_log_edges[class_2d])
        assert (np.log(v1) < v_log_edges[class_2d + 1])

    # test find_3d_class_number
    for i in range(mapping.n_2d_classes):
        class_2d = i
        cumsum_n_subclass = mapping.cumsum_n_subclass
        freq_array = mapping.sub_classes_nrepeat[class_2d]
        index_2d = class_2d * np.ones(len(freq_array), dtype=np.int)
        class_3d = mapping.find_3d_class_number(index_2d, freq_array)
        assert (np.all(class_3d == (cumsum_n_subclass[index_2d] + range(len(freq_array)))))
        freq_array_2 = np.zeros(len(class_3d))
        for j in range(len(class_3d)):
            v, theta, freq = mapping.find_v_theta_freq(class_3d[j])
            assert (freq == freq_array[j])
            inv_class_2d = mapping.class_index_2d_vtheta(v, theta)
            assert (inv_class_2d == class_2d)

def test_mapping_theta_2():
    main_folder = os.path.dirname(os.path.dirname(__file__))
    input_folder = os.path.join(main_folder, 'test_related_files', 'particle_tracking_results')
    dt = 50.0
    n_realz = 1
    v_array, theta_array, freq_array, pointer_list, initial_v, initial_f, initial_theta = make_input_for_binning_v_theta_freq(
        input_folder, n_realz, dt)
    new_v, new_theta, new_f = remove_duplicate_xy(v_array, theta_array, freq_array)
    # make random angles
    n_abs_log_class = 100
    n_theta_classes = 60
    abs_log_v_edges = abs_vel_log_bins_low_high(new_v, n_abs_log_class, n_low=5, max_allowed=0.03)
    theta_bin_edges = make_theta_bins_linear(n_theta_classes)
    mapping = mapping_v_theta_repeat(abs_log_v_edges, theta_bin_edges, new_v, new_theta, new_f)
    v0_idx = mapping.find_1d_class_idx(np.log(new_v), mapping.v_log_edges)
    t0_idx = mapping.find_1d_class_idx(new_theta, mapping.theta_edges)
    # find the 3d class number for the input
    class_2d = mapping.class_index_2d_vtheta(new_v, new_theta)
    class_3d = mapping.find_3d_class_number(class_2d, new_f)
    # convert back from 2d class and check indices
    v1_idx, t1_idx = mapping.class_index_1d_v_theta_from_2d(class_2d)
    assert (np.all(v1_idx == v0_idx))
    assert (np.all(t1_idx == t0_idx))
    # convert back from 3d and check the indics
    v2 = np.zeros(len(class_3d))
    t2 = np.zeros(len(class_3d))
    f2 = np.zeros(len(class_3d))
    for i in range(len(class_3d)):
        v2[i], t2[i], f2[i] = mapping.find_v_theta_freq(class_3d[i])
    v2_idx = mapping.find_1d_class_idx(np.log(v2), mapping.v_log_edges)
    t2_idx = mapping.find_1d_class_idx(t2, mapping.theta_edges)
    assert (np.all(v2_idx == v0_idx))
    assert (np.all(t2_idx == t0_idx))
    assert (np.all(f2 == new_f))