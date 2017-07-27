from py_dp.dispersion.binning import make_input_for_binning_with_freq, make_1d_abs_vel_bins, class_index_abs_log
from py_dp.dispersion.convert_to_time_process_with_freq import remove_duplicate
import numpy as np
from copy import copy
from py_dp.dispersion.mapping import mapping_v_sgn_repeat
import os

def test_mapping_both_ways():
    main_folder = os.path.dirname(os.path.dirname(__file__))
    input_folder = os.path.join(main_folder, 'test_related_files', 'particle_tracking_results')
    dt = 50.0
    n_realz = 1
    big_v_array, big_freq_array, pointer_list, initial_v0, initial_v1 = make_input_for_binning_with_freq(input_folder,
                                                                                                         n_realz, dt)
    new_v, new_f = remove_duplicate(big_v_array, big_freq_array)
    n_abs_log_class = 8
    abs_log_v_edges = make_1d_abs_vel_bins(new_v, n_abs_log_class, n_slow_classes = 1)
    v_class_number = class_index_abs_log(new_v, abs_log_v_edges)

    sub_classes_nrepeat = []
    n_subclass = []
    place_holder = np.array([1], dtype=np.int)
    for i in range(2*n_abs_log_class):
        possible_f_vals = np.unique(new_f[v_class_number == i])
        if not len(possible_f_vals):
            possible_f_vals = copy(place_holder)
        sub_classes_nrepeat.append(sorted(possible_f_vals))
        n_subclass.append(len(possible_f_vals))
    modified_n_sub_class = np.array(n_subclass)
    cumsum_n_subclass = np.hstack((0,np.cumsum(modified_n_sub_class)))

    mapping = mapping_v_sgn_repeat(abs_log_v_edges, cumsum_n_subclass, sub_classes_nrepeat)

    #test draw velocity
    v_log_edges = mapping.v_log_edges
    # for i in range(len(v_log_edges)-1):
    for i in range(mapping.n_abs_v_classes):
        class_2d = i
        v1 = mapping.draw_from_class_velocity(class_2d)
        v_log_edges = mapping.v_log_edges
        assert(np.log(v1)>v_log_edges[class_2d])
        assert(np.log(v1)<v_log_edges[class_2d+1])

    #test find_3d_class_number
    for i in range(len(v_log_edges)-1):
        class_2d = i
        cumsum_n_subclass = mapping.cumsum_n_subclass
        freq_array = mapping.sub_classes_nrepeat[class_2d]
        index_2d = class_2d*np.ones(len(freq_array), dtype=np.int)
        class_3d = mapping.find_3d_class_number(index_2d, freq_array)
        assert(np.all(class_3d == (cumsum_n_subclass[index_2d] + range(len(freq_array)))))

    # #test find_3d_class for freq values not available in the binning data
    # index_2d_test = [0, 0, 0]
    # freq_test = [15.0, 90.0, 125]
    # test_output = mapping.find_3d_class_number(index_2d_test, freq_test)
    # print test_output
    # assert(np.all(test_output == [13, 24, 24]))
    # #test for last class
    # index_2d_test = (mapping.n_2d_classes - 1)*np.ones(3, dtype=int)
    # freq_test = [15.0, 90.0, 125]
    # test_output = mapping.find_3d_class_number(index_2d_test, freq_test)
    # expected_output = (mapping.n_3d_classes-1)*np.ones(3, dtype=int)
    # assert(np.all(test_output == expected_output))
    #
    # #test inverse mapping
    # class_3d_array = range(mapping.n_3d_classes)
    # for class_3d_test in class_3d_array:
    #     abs_v, sgn_v, freq = mapping.find_v_sgn_freq(class_3d_test)


