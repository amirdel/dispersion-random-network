import numpy as np
import pickle
import os
from copy import copy
from scipy.sparse import csc_matrix
from py_dp.dispersion.binning import make_1d_abs_vel_bins
from py_dp.dispersion.trajectory_count_cython import count_with_lag_one_trajectory_kang_cython
from py_dp.dispersion.trajectory_count_cython import count_with_lag_one_trajectory_aggr_cython
from py_dp.dispersion.binning import make_input_for_binning_no_freq
from py_dp.dispersion.binning import make_input_for_binning_with_freq
from py_dp.dispersion.transition_matrix_fcns import normalize_columns
from py_dp.dispersion.second_order_markov import find_2d_bin
from py_dp.dispersion.binning import class_index_abs_log
from py_dp.dispersion.trajectory_count_cython import fill_2nd_order_one_trajectory_cython
from py_dp.dispersion.binning import fix_out_of_bound
from py_dp.dispersion.transition_matrix_fcns import count_matrix_with_freq_one_trajectory_kang
from py_dp.dispersion.transition_matrix_fcns import count_matrix_with_freq_one_trajectory_agg
from py_dp.dispersion.convert_to_time_process_with_freq import get_time_dx_array_with_frequency
from py_dp.dispersion.mapping import mapping_v_sgn_repeat
from py_dp.dispersion.convert_to_time_process_with_freq import remove_duplicate
from py_dp.dispersion.mapping import mapping_v_theta_repeat
from py_dp.dispersion.binning import abs_vel_log_bins_low_high, make_theta_bins_linear
from py_dp.dispersion.binning import make_input_for_binning_v_theta_freq_with_filter
from py_dp.dispersion.convert_to_time_process_with_freq import remove_duplicate_xy
from py_dp.dispersion.trajectory_count_cython import fill_one_trajectory_sparse_cython, fill_one_trajectory_sparse_with_freq_cython
from py_dp.dispersion.convert_to_time_process_with_freq import get_time_dx_dy_array_with_freq


class GenerateTransitionInfoKang(object):
    def __init__(self, input_folder, n_binning_realz, n_total_realz, n_class, n_slow_class = 2):
        '''
        Here we create bins from the first n_binning_realizations, keep the data that we read
        to use for filling the trasition matrix from the same realizations
        '''
        self.input_folder = input_folder
        self.n_binning_realz = n_binning_realz
        self.n_total_realz = n_total_realz
        self.n_class = n_class
        self.n_slow_class = n_slow_class
        self.make_velocity_bins = make_1d_abs_vel_bins
        self.big_v_array, self.pointer_list, self.initial_v0 , self.initial_v1 = make_input_for_binning_no_freq(input_folder, n_binning_realz)
        self.v_log_edges = self.make_bin_data()
        self.o1_init_class_count = self.get_o1_init_class_count()
        self.o2_init_class_count = self.get_o2_init_class_count()

    def make_bin_data(self):
        print "making bins..."
        v_log_edges = self.make_velocity_bins(self.big_v_array, self.n_class, self.n_slow_class)
        print "done."
        return v_log_edges

    def get_o1_init_class_count(self):
        #here we are getting class for value of veloccity, we know the direction
        initial_v0_class = np.digitize(np.log(np.abs(self.initial_v0)), self.v_log_edges)
        fix_out_of_bound(initial_v0_class, self.v_log_edges)
        initial_v0_class -= 1
        init_class_count = np.zeros(self.n_class)
        for i in initial_v0_class:
            init_class_count[i] += 1
        return init_class_count

    def get_o2_init_class_count(self):
        n_1st_order_bins = 2*self.n_class
        initial_v0_class = class_index_abs_log(self.initial_v0, self.v_log_edges)
        initial_v1_class = class_index_abs_log(self.initial_v1, self.v_log_edges)
        init_class_count = np.zeros(n_1st_order_bins*n_1st_order_bins)
        for i in range(len(initial_v0_class)):
            class_number_2d = find_2d_bin(initial_v0_class[i], initial_v1_class[i], n_1st_order_bins)
            init_class_count[class_number_2d] += 1
        return init_class_count

    def get_trans_matrix_with_lag(self, lag):
        dim_size = 2*self.n_class + 1
        v_log_edges = self.v_log_edges
        pointer_list = self.pointer_list
        v_array = self.big_v_array
        transition_count_matrix = np.zeros((dim_size,dim_size))
        start_idx = 0
        for i in pointer_list:
            v_temp = v_array[start_idx:i]
            start_idx = i
            count_with_lag_one_trajectory_kang_cython(transition_count_matrix, lag, v_temp, v_log_edges)
        for j in range(self.n_binning_realz, self.n_total_realz):
            file_name = "real_"+str(j)+".pkl";
            input_file = os.path.join(self.input_folder, file_name)
            with open(input_file,'rb') as input:
                dataHolder = pickle.load(input)
            dx = np.diff(dataHolder.x_array)
            dt = np.diff(dataHolder.t_array)
            lastIdx = dataHolder.last_idx_array
            vMatrix = np.divide(dx,dt)
            m = dx.shape[0]
            for i in range(m):
                cutOff = lastIdx[i]
                v_temp = vMatrix[i, 0:cutOff - 1]
                count_with_lag_one_trajectory_kang_cython(transition_count_matrix, lag, v_temp, v_log_edges)
        transition_matrix = normalize_columns(transition_count_matrix)
        return transition_matrix

    def get_trans_matrix_aggregate(self, lag):
        dim_size = self.n_class
        v_log_edges = self.v_log_edges
        pointer_list = self.pointer_list
        v_array = self.big_v_array
        transition_count_matrix = np.zeros((dim_size,dim_size))
        start_idx = 0
        print 'extracting trans matrix...'
        for i in pointer_list:
            v_temp = v_array[start_idx:i]
            start_idx = i
            count_with_lag_one_trajectory_aggr_cython(transition_count_matrix, lag, v_temp, v_log_edges)
        for j in range(self.n_binning_realz, self.n_total_realz):
            print 'realization number: ',j
            file_name = "real_"+str(j)+".pkl";
            input_file = os.path.join(self.input_folder, file_name)
            with open(input_file,'rb') as input:
                dataHolder = pickle.load(input)
            dx = np.diff(dataHolder.x_array)
            dt = np.diff(dataHolder.t_array)
            lastIdx = dataHolder.last_idx_array
            vMatrix = np.divide(dx,dt)
            m = dx.shape[0]
            for i in range(m):
                cutOff = lastIdx[i]
                v_temp = vMatrix[i, 0:cutOff - 1]
                count_with_lag_one_trajectory_aggr_cython(transition_count_matrix, lag, v_temp, v_log_edges)
        transition_matrix_aggregate = normalize_columns(transition_count_matrix)
        return transition_matrix_aggregate

    def get_2nd_markov_sparse(self, lag):
        pointer_list = self.pointer_list
        v_array = self.big_v_array
        v_log_edges = self.v_log_edges
        #n_class is the number of classes for abs(v)
        n_class = self.n_class
        #n_abs_vel_class is the number of (abs(v), sgn(v)) classes used in Kang et al.
        n_abs_vel_class = 2*n_class
        n_class_square = n_abs_vel_class*n_abs_vel_class
        i_list = []
        j_list = []
        ij_list = set([])
        val_list = []
        start_idx = 0
        print 'extracting trans matrix...'
        for i in pointer_list:
            v_temp = v_array[start_idx:i]
            start_idx = i
            fill_2nd_order_one_trajectory_cython(lag, v_temp, v_log_edges, i_list, j_list, ij_list,
                                          val_list, n_abs_vel_class)
        for j in range(self.n_binning_realz, self.n_total_realz):
            print 'realization number: ',j
            file_name = "real_"+str(j)+".pkl";
            input_file = os.path.join(self.input_folder, file_name)
            with open(input_file,'rb') as input:
                dataHolder = pickle.load(input)
            dx = np.diff(dataHolder.x_array)
            dt = np.diff(dataHolder.t_array)
            lastIdx = dataHolder.last_idx_array
            vMatrix = np.divide(dx,dt)
            m = dx.shape[0]
            for i in range(m):
                cutOff = lastIdx[i]
                v_temp = vMatrix[i, 0:cutOff - 1]
                fill_2nd_order_one_trajectory_cython(lag, v_temp, v_log_edges, i_list, j_list, ij_list,
                                              val_list, n_abs_vel_class)
        return csc_matrix((val_list, (i_list, j_list)), shape = (n_class_square, n_class_square))

class GenerateTransitionInfoTime(object):
    def __init__(self, input_folder, n_binning_realz, n_total_realz, n_class, time_step, n_slow_class = 1):
        '''
        Here we create bins from the first n_binning_realizations, keep the data that we read
        to use for filling the trasition matrix from the same realizations
        '''
        self.time_step = time_step
        self.input_folder = input_folder
        self.n_binning_realz = n_binning_realz
        self.n_total_realz = n_total_realz
        self.n_class = n_class
        self.n_slow_class = n_slow_class
        self.make_velocity_bins = make_1d_abs_vel_bins
        self.big_v_array, self.big_freq_array, self.pointer_list, self.initial_v0 , self.initial_v1 = make_input_for_binning_with_freq(input_folder, n_binning_realz, time_step)
        self.v_log_edges = self.make_bin_data()
        self.o1_init_class_count = self.get_o1_init_class_count()
        #self.o2_init_class_count = self.get_o2_init_class_count()

    def make_bin_data(self):
        print "making bins..."
        v_log_edges = self.make_velocity_bins(self.big_v_array, self.n_class, self.n_slow_class)
        print "done."
        return v_log_edges

    def get_o1_init_class_count(self):
        #here we are getting class for value of veloccity, we know the direction
        initial_v0_class = np.digitize(np.log(np.abs(self.initial_v0)), self.v_log_edges)
        fix_out_of_bound(initial_v0_class, self.v_log_edges)
        initial_v0_class -= 1
        init_class_count = np.zeros(self.n_class)
        for i in initial_v0_class:
            init_class_count[i] += 1
        return init_class_count

    # def get_o2_init_class_count(self):
    #     n_1st_order_bins = 2*self.n_class
    #     initial_v0_class = class_index_abs_log(self.initial_v0, self.v_log_edges)
    #     initial_v1_class = class_index_abs_log(self.initial_v1, self.v_log_edges)
    #     init_class_count = np.zeros(n_1st_order_bins*n_1st_order_bins)
    #     for i in range(len(initial_v0_class)):
    #         class_number_2d = find_2d_bin(initial_v0_class[i], initial_v1_class[i], n_1st_order_bins)
    #         init_class_count[class_number_2d] += 1
    #     return init_class_count

    def get_trans_matrix_with_lag(self, lag):
        print 'extracting trans matrix...'
        time_step = self.time_step
        dim_size = 2*self.n_class + 1
        v_log_edges = self.v_log_edges
        pointer_list = self.pointer_list
        v_array = self.big_v_array
        freq_array = self.big_freq_array
        transition_count_matrix = np.zeros((dim_size,dim_size))
        start_idx = 0
        for i in pointer_list:
            v_temp = v_array[start_idx:i]
            freq_temp = freq_array[start_idx:i]
            start_idx = i
            count_matrix_with_freq_one_trajectory_kang(transition_count_matrix, lag, v_temp, freq_temp, v_log_edges)
        for j in range(self.n_binning_realz, self.n_total_realz):
            print 'realization number: ',j
            file_name = "real_"+str(j)+".pkl";
            input_file = os.path.join(self.input_folder, file_name)
            with open(input_file,'rb') as input:
                dataHolder = pickle.load(input)
            dx = np.diff(dataHolder.x_array)
            dt = np.diff(dataHolder.t_array)
            lastIdx = dataHolder.last_idx_array
            vMatrix = np.divide(dx,dt)
            m = dx.shape[0]
            for i in range(m):
                cutOff = lastIdx[i]
                dx_time, freq_temp = get_time_dx_array_with_frequency(dt[i,:cutOff], vMatrix[i,:cutOff], time_step)
                v_temp = np.array(dx_time)/time_step
                count_matrix_with_freq_one_trajectory_kang(transition_count_matrix, lag, v_temp, freq_temp, v_log_edges)
        transition_matrix = normalize_columns(transition_count_matrix)
        print 'done'
        return transition_matrix

    def get_trans_matrix_aggregate(self, lag):
        print 'extracting trans matrix...'
        time_step = self.time_step
        dim_size = self.n_class
        v_log_edges = self.v_log_edges
        pointer_list = self.pointer_list
        v_array = self.big_v_array
        freq_array = self.big_freq_array
        transition_count_matrix = np.zeros((dim_size,dim_size))
        start_idx = 0
        for i in pointer_list:
            v_temp = v_array[start_idx:i]
            freq_temp = freq_array[start_idx:i]
            start_idx = i
            count_matrix_with_freq_one_trajectory_agg(transition_count_matrix, lag, v_temp, freq_temp, v_log_edges)
        for j in range(self.n_binning_realz, self.n_total_realz):
            print 'realization number: ',j
            file_name = "real_"+str(j)+".pkl"
            input_file = os.path.join(self.input_folder, file_name)
            with open(input_file,'rb') as input:
                dataHolder = pickle.load(input)
            dx = np.diff(dataHolder.x_array)
            dt = np.diff(dataHolder.t_array)
            lastIdx = dataHolder.last_idx_array
            vMatrix = np.divide(dx,dt)
            m = dx.shape[0]
            for i in range(m):
                cutOff = lastIdx[i]
                dx_time, freq_temp = get_time_dx_array_with_frequency(dt[i,:cutOff], vMatrix[i,:cutOff], time_step)
                v_temp = np.array(dx_time)/time_step
                count_matrix_with_freq_one_trajectory_agg(transition_count_matrix, lag, v_temp, freq_temp, v_log_edges)
        transition_matrix = normalize_columns(transition_count_matrix)
        print 'done'
        return transition_matrix


class GenerateTransitionInfo3d(object):
    def __init__(self, input_folder, n_binning_realz, n_total_realz, n_absv_class, time_step, n_slow_class = 1):
        '''
        Here we create bins from the first n_binning_realizations, keep the data that we read
        to use for filling the trasition matrix from the same realizations
        '''
        self.time_step = time_step
        self.input_folder = input_folder
        self.n_binning_realz = n_binning_realz
        self.n_total_realz = n_total_realz
        self.n_absv_class = n_absv_class
        self.n_slow_class = n_slow_class
        self.make_velocity_bins = make_1d_abs_vel_bins
        self.big_v_array, self.big_freq_array, self.pointer_list, self.initial_v0 , self.initial_f0 = \
        make_input_for_binning_with_freq(input_folder, n_binning_realz, time_step)

        self.v_log_edges = self.make_bin_data()
        self.mapping = self.generate_mapping()
        self.init_class_count = self.get_init_class_count()

        #self.o2_init_class_count = self.get_o2_init_class_count()

    def make_bin_data(self):
        print "making bins..."
        v_log_edges = self.make_velocity_bins(self.big_v_array, self.n_absv_class, self.n_slow_class)
        print "done."
        return v_log_edges

    def generate_mapping(self):
        '''
        :param self:
        :return: mapping
        '''
        print "generating map..."
        new_v, new_f = remove_duplicate(self.big_v_array, self.big_freq_array)
        v_log_edges = self.v_log_edges
        v_class_number = class_index_abs_log(new_v, v_log_edges)
        sub_classes_nrepeat = []
        n_subclass = []
        place_holder = np.array([1], dtype=np.int)
        for i in range(2*self.n_absv_class):
            possible_f_vals = np.unique(new_f[v_class_number == i])
            if not len(possible_f_vals):
                possible_f_vals = copy(place_holder)
            sub_classes_nrepeat.append(sorted(possible_f_vals))
            n_subclass.append(len(possible_f_vals))
        modified_n_sub_class = np.array(n_subclass)
        cumsum_n_subclass = np.hstack((0,np.cumsum(modified_n_sub_class)))
        mapping = mapping_v_sgn_repeat(v_log_edges, cumsum_n_subclass, sub_classes_nrepeat)
        print "done"
        return mapping

    def get_init_class_count(self):
        new_v, new_f = remove_duplicate(self.initial_v0, self.initial_f0)
        v_log_edges = self.v_log_edges
        v_class_number = class_index_abs_log(new_v, v_log_edges)
        init_count = self.mapping.find_3d_class_number(v_class_number, new_f)
        init_class_count = np.zeros(self.mapping.n_3d_classes)
        for i in init_count:
            init_class_count[i] += 1
        return init_class_count

    def get_trans_matrix(self, lag):
        n_3d_class = self.mapping.n_3d_classes
        v_log_edges = self.v_log_edges
        i_list = []
        j_list = []
        ij_list = set([])
        val_list = []
        time_step = self.time_step
        print 'extracting trans matrix...'
        for j in range(self.n_total_realz):
            print 'realization number: ',j
            file_name = "real_"+str(j)+".pkl"
            input_file = os.path.join(self.input_folder, file_name)
            with open(input_file,'rb') as input:
                dataHolder = pickle.load(input)
            dx = np.diff(dataHolder.x_array)
            dt = np.diff(dataHolder.t_array)
            lastIdx = dataHolder.last_idx_array
            vMatrix = np.divide(dx,dt)
            m = dx.shape[0]
            assert (np.all(vMatrix[:, 0] > 0.0))
            for i in range(m):
                # get the time process for each velocity
                cutOff = lastIdx[i]
                dx_time, freq_temp = get_time_dx_array_with_frequency(dt[i, :cutOff], vMatrix[i, :cutOff], time_step)
                v_temp = np.array(dx_time)/time_step
                if len(dx_time) > 1:
                    new_v, new_f = remove_duplicate(v_temp, freq_temp)
                    #get the velocity class number from abs(v) and sgn(v)
                    v_class_number = class_index_abs_log(new_v, v_log_edges)
                    class_3d_array = self.mapping.find_3d_class_number(v_class_number, new_f)
                    fill_one_trajectory_sparse_cython(lag, class_3d_array, i_list, j_list, ij_list, val_list)
        print 'done'
        return csc_matrix((val_list, (i_list, j_list)), shape = (n_3d_class, n_3d_class))

class GenerateTransitionInfo_xy_2d_filtered(object):
    """
    a class for extracting binned trasition information for 2d spatial cases with 2d bins (log|v|, theta)
    """
    def __init__(self, input_folder, n_binning_realz, n_total_realz, n_absv_class,
                 n_theta_class, time_step, n_slow_class=1, max_allowed=0.03, filter_length=None, filter_time=None):
        '''
        Transition matrix for 3d classes describing v, theta, frequency
        '''
        self.time_step = time_step
        self.input_folder = input_folder
        self.n_binning_realz = n_binning_realz
        self.n_total_realz = n_total_realz
        self.n_absv_class = n_absv_class
        self.n_theta_class = n_theta_class
        self.n_slow_class = n_slow_class
        self.max_allowed = max_allowed
        # self.make_velocity_bins = make_1d_abs_vel_bins
        self.make_velocity_bins = abs_vel_log_bins_low_high
        self.make_theta_bins = make_theta_bins_linear
        # making sample data for creating bins
        self.big_v_array, self.big_theta_array, self.big_freq_array, self.pointer_list, self.initial_v, self.initial_f, self.initial_theta = \
            make_input_for_binning_v_theta_freq_with_filter(input_folder, n_binning_realz, time_step, filter_time)
        self.v_log_edges, self.theta_edges = self.make_bin_data()
        self.mapping = self.generate_mapping()
        self.init_class_count = self.get_init_class_count()
        self.filter_length = filter_length
        self.filter_time = filter_time
        # self.o2_init_class_count = self.get_o2_init_class_count()

    def make_bin_data(self):
        print "making v bins..."
        # v_log_edges = self.make_velocity_bins(self.big_v_array, self.n_absv_class, self.n_slow_class)
        v_log_edges = self.make_velocity_bins(self.big_v_array, self.n_absv_class,
                                              max_allowed=self.max_allowed)
        print "done."
        print 'making theta bins...'
        theta_edges = self.make_theta_bins(self.n_theta_class)
        print 'done'
        return v_log_edges, theta_edges

    def generate_mapping(self):
        '''
        :param self:
        :return: mapping
        '''
        print "generating map..."
        mapping = mapping_v_theta_repeat(self.v_log_edges, self.theta_edges, self.big_v_array, self.big_theta_array,
                                         self.big_freq_array,
                                         make_aux_arrays=False)
        print "done"
        return mapping

    def get_init_class_count(self):
        new_v, new_theta, new_f = remove_duplicate_xy(self.initial_v, self.initial_theta, self.initial_f)
        index_2d = self.mapping.class_index_2d_vtheta(new_v, new_theta)
        init_class_count = np.zeros(self.mapping.n_2d_classes)
        for i in index_2d:
            init_class_count[i] += 1
        return init_class_count

    def get_trans_matrix(self, lag):
        filter_length = self.filter_length
        if filter_length:
            print 'using only points with x values less than '+ str(filter_length)
        filter_time = self.filter_time
        if filter_time:
            print 'using only points with time less than ' + str(filter_time)
        n_2d_class = self.mapping.n_2d_classes
        i_list = []
        j_list = []
        ij_list = set([])
        val_list = []
        time_step = self.time_step
        print 'extracting trans matrix...'
        for j in range(self.n_total_realz):
            print 'realization number: ', j
            file_name = "real_" + str(j) + ".pkl"
            input_file = os.path.join(self.input_folder, file_name)
            with open(input_file, 'rb') as input:
                dataHolder = pickle.load(input)
            dx = np.diff(dataHolder.x_array)
            dy = np.diff(dataHolder.y_array)
            dt = np.diff(dataHolder.t_array) + 1e-15
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
                if filter_length:
                    cutOff = min(cutOff, np.argmin(dataHolder.x_array < filter_length))
                if filter_time:
                    cutOff = min(cutOff, np.argmin(dataHolder.t_array < filter_time))
                dxTime, dyTime, freq = get_time_dx_dy_array_with_freq(dt[i, :cutOff], vxMatrix[i, :cutOff],
                                                                      vyMatrix[i, :cutOff], x_start, y_start,
                                                                      time_step)
                v_temp = np.sqrt(np.power(dxTime, 2) + np.power(dyTime, 2)) / time_step
                theta_temp = np.arctan2(dyTime, dxTime)
                if len(v_temp) > lag:
                    new_v, new_theta, new_f = remove_duplicate_xy(v_temp, theta_temp, freq)
                    class_2d = self.mapping.class_index_2d_vtheta(new_v, new_theta)
                    new_f = np.array(new_f, dtype=np.dtype("i"))
                    fill_one_trajectory_sparse_with_freq_cython(lag, class_2d, new_f, i_list, j_list, ij_list,
                                                                val_list)
        print 'done'
        return csc_matrix((val_list, (i_list, j_list)), shape=(n_2d_class, n_2d_class))