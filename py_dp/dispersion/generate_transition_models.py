import numpy as np
import pickle
import os
from scipy.sparse import csc_matrix
from py_dp.dispersion.convert_to_time_process_with_freq import get_time_dx_dy_array_with_freq
from py_dp.dispersion.trajectory_count_cython import fill_one_trajectory_sparse_cython, fill_one_trajectory_sparse_with_freq_cython
from py_dp.dispersion.convert_to_time_process_with_freq import remove_duplicate_xy

class TransitionInfoExtendedStencil(object):
    def __init__(self, input_folder, n_total_realz, mapping, mapping_input, average_available=False):
        """
        Object for generating transition matrix and the initial class count for the two dimensional extended stencil method
        :param input_folder: input folder containing the MC data
        :param n_total_realz: number of realizations to use for extracting matrices
        :param mapping: mapping object for the extended stencil method
        :param mapping_input: mapping_input object containing the example trajectories for estimating the distribution
                              of observed classes for initializing the simulations
        """
        self.time_step = mapping_input.time_step
        self.input_folder = input_folder
        self.n_total_realz = n_total_realz
        self.v_log_edges, self.theta_edges = mapping_input.v_log_edges, mapping_input.theta_edges
        self.mapping = mapping
        self.init_class_count = self.get_init_class_count(mapping_input)
        self.average_available = average_available

    def get_init_class_count(self, map_input):
        new_v, new_theta, new_f = remove_duplicate_xy(map_input.initial_v, map_input.initial_theta, map_input.initial_f)
        index_2d = self.mapping.class_index_2d_vtheta(new_v, new_theta)
        index_3d = self.mapping.find_3d_class_number(index_2d, new_f)
        init_class_count = np.zeros(self.mapping.n_3d_classes)
        for i in index_3d:
            init_class_count[i] += 1
        return init_class_count

    def get_trans_matrix(self, lag, print_every = 50, verbose=True):
        if self.average_available:
            return self.get_trans_matrix_from_average(lag, print_every, verbose)
        else:
            return self.get_trans_matrix_from_scratch(lag, print_every, verbose)

    def get_trans_matrix_from_average(self, lag, print_every=1, verbose=True):
        n_3d_class = self.mapping.n_3d_classes
        i_list, j_list, val_list  = [[] for i in range(3)]
        ij_list = set([])
        for j in range(self.n_total_realz):
            start_idx = 0
            # load the polar coordinates file
            data_path = os.path.join(self.input_folder, 'avg_polar_' + str(j) + '.npz')
            data = np.load(data_path)
            big_v, big_theta, big_f, ptr_list = data['V'], data['Theta'], data['F'], data['ptr']
            for i in ptr_list:
                new_v, new_theta, new_f = big_v[start_idx:i], big_theta[start_idx:i], big_f[start_idx:i]
                start_idx = i
                if len(new_v) > lag:
                    class_2d = self.mapping.class_index_2d_vtheta(new_v, new_theta)
                    class_3d_array = self.mapping.find_3d_class_number(class_2d, new_f)
                    fill_one_trajectory_sparse_cython(lag, class_3d_array, i_list, j_list, ij_list, val_list)
        print 'done'
        return csc_matrix((val_list, (i_list, j_list)), shape=(n_3d_class, n_3d_class))

    def get_trans_matrix_from_scratch(self, lag, print_every=50, verbose=True):
        n_3d_class = self.mapping.n_3d_classes
        i_list = []
        j_list = []
        ij_list = set([])
        val_list = []
        time_step = self.time_step
        print 'extracting trans matrix...'
        for j in range(self.n_total_realz):
            if verbose and not j%print_every:
                print 'reading realization number: ',j
            file_name = "real_"+str(j)+".pkl"
            input_file = os.path.join(self.input_folder, file_name)
            with open(input_file,'rb') as input:
                dataHolder = pickle.load(input)
            dx = np.diff(dataHolder.x_array)
            dy = np.diff(dataHolder.y_array)
            dt = np.diff(dataHolder.t_array) + 1e-15
            if not (dx.shape[0] and dy.shape[0] and dt.shape[0]):
                print 'some array was empty, skipping this file...'
                continue
            lastIdx = dataHolder.last_idx_array
            vxMatrix = np.divide(dx,dt)
            vyMatrix = np.divide(dy,dt)
            m = dx.shape[0]
            for i in range(m):
                x_start = dataHolder.x_array[i, 0]
                y_start = dataHolder.y_array[i, 0]
                # get the time process for each velocity
                cutOff = lastIdx[i]
                dxTime, dyTime, freq = get_time_dx_dy_array_with_freq(dt[i, :cutOff], vxMatrix[i, :cutOff],
                                                                      vyMatrix[i, :cutOff], x_start, y_start, time_step)
                v_temp = np.sqrt(np.power(dxTime,2) + np.power(dyTime,2))/time_step
                theta_temp = np.arctan2(dyTime, dxTime)
                if len(v_temp) > lag:
                    new_v, new_theta, new_f = remove_duplicate_xy(v_temp, theta_temp, freq)
                    class_2d = self.mapping.class_index_2d_vtheta(new_v, new_theta)
                    class_3d_array = self.mapping.find_3d_class_number(class_2d, new_f)
                    fill_one_trajectory_sparse_cython(lag, class_3d_array, i_list, j_list, ij_list, val_list)
        print 'done'
        return csc_matrix((val_list, (i_list, j_list)), shape = (n_3d_class, n_3d_class))


class TransitionInfoStencilMethod(object):
    """
    a class for extracting binned trasition information for 2d spatial cases with 2d bins (log|v|, theta)
    """
    def __init__(self, input_folder, n_total_realz, mapping, mapping_input, average_available=False):
        """
        Object for generating transition matrix and the initial class count for the two dimensional extended stencil method
        :param input_folder: input folder containing the MC data
        :param n_total_realz: number of realizations to use for extracting matrices
        :param mapping: mapping object for the extended stencil method
        :param mapping_input: mapping_input object containing the example trajectories for estimating the distribution
                              of observed classes for initializing the simulations
        """
        self.time_step = mapping_input.time_step
        self.input_folder = input_folder
        self.n_total_realz = n_total_realz
        self.v_log_edges, self.theta_edges = mapping_input.v_log_edges, mapping_input.theta_edges
        self.mapping = mapping
        self.init_class_count = self.get_init_class_count(mapping_input)
        self.average_available = average_available

    def get_init_class_count(self, map_input):
        new_v, new_theta, new_f = remove_duplicate_xy(map_input.initial_v, map_input.initial_theta, map_input.initial_f)
        index_2d = self.mapping.class_index_2d_vtheta(new_v, new_theta)
        init_class_count = np.zeros(self.mapping.n_2d_classes)
        for i in index_2d:
            init_class_count[i] += 1
        return init_class_count

    def get_trans_matrix(self, lag, print_every = 50, verbose=True):
        if self.average_available:
            return self.get_trans_matrix_from_average(lag, print_every, verbose)
        else:
            return self.get_trans_matrix_from_scratch(lag, print_every, verbose)

    def get_trans_matrix_from_scratch(self, lag, print_every = 50, verbose=True):
        n_2d_class = self.mapping.n_2d_classes
        i_list = []
        j_list = []
        ij_list = set([])
        val_list = []
        time_step = self.time_step
        print 'extracting trans matrix...'
        for j in range(self.n_total_realz):
            if verbose and not j%print_every:
                print 'reading realization number: ',j
            file_name = "real_"+str(j)+".pkl"
            input_file = os.path.join(self.input_folder, file_name)
            with open(input_file,'rb') as input:
                dataHolder = pickle.load(input)
            dx = np.diff(dataHolder.x_array)
            dy = np.diff(dataHolder.y_array)
            dt = np.diff(dataHolder.t_array) + 1e-15
            if not (dx.shape[0] and dy.shape[0] and dt.shape[0]):
                print 'some array was empty, skipping this file...'
                continue
            lastIdx = dataHolder.last_idx_array
            vxMatrix = np.divide(dx,dt)
            vyMatrix = np.divide(dy,dt)
            m = dx.shape[0]
            for i in range(m):
                x_start = dataHolder.x_array[i, 0]
                y_start = dataHolder.y_array[i, 0]
                # get the time process for each velocity
                cutOff = lastIdx[i]
                dxTime, dyTime, freq = get_time_dx_dy_array_with_freq(dt[i, :cutOff], vxMatrix[i, :cutOff],
                                                                      vyMatrix[i, :cutOff], x_start, y_start, time_step)
                v_temp = np.sqrt(np.power(dxTime,2) + np.power(dyTime,2))/time_step
                theta_temp = np.arctan2(dyTime, dxTime)
                if len(v_temp) > lag:
                    new_v, new_theta, new_f = remove_duplicate_xy(v_temp, theta_temp, freq)
                    class_2d = self.mapping.class_index_2d_vtheta(new_v, new_theta)
                    fill_one_trajectory_sparse_with_freq_cython(lag, class_2d, new_f, i_list, j_list, ij_list, val_list)
        print 'done'
        return csc_matrix((val_list, (i_list, j_list)), shape = (n_2d_class, n_2d_class))

    def get_trans_matrix_from_average(self, lag, print_every = 50, verbose=True):
        n_2d_class = self.mapping.n_2d_classes
        i_list = []
        j_list = []
        ij_list = set([])
        val_list = []
        for j in range(self.n_total_realz):
            start_idx = 0
            # load the polar coordinates file
            data_path = os.path.join(self.input_folder, 'avg_polar_'+str(j)+'.npz')
            data = np.load(data_path)
            big_v, big_theta, big_f, ptr_list = data['V'], data['Theta'], data['F'], data['ptr']
            for i in ptr_list:
                new_v, new_theta, new_f = big_v[start_idx:i], big_theta[start_idx:i], big_f[start_idx:i]
                start_idx = i
                if len(new_v) > lag:
                    class_2d = self.mapping.class_index_2d_vtheta(new_v, new_theta)
                    fill_one_trajectory_sparse_with_freq_cython(lag, class_2d, new_f, i_list, j_list, ij_list, val_list)
        print 'done'
        return csc_matrix((val_list, (i_list, j_list)), shape = (n_2d_class, n_2d_class))

class TransitionInfoSpace(object):
    def __init__(self, input_folder, n_total_realz, mapping, mapping_input):
        """
        Object for generating transition matrix and the initial class count for the spatial Markov model
        :param input_folder: input folder containing the MC data
        :param n_total_realz: number of realizations to use for extracting matrices
        :param mapping: mapping object for the spacial Markov model
        :param mapping_input: mapping_input object containing the example trajectories for estimating the distribution
                              of observed classes for initializing the simulations
        """
        self.input_folder = input_folder
        self.n_total_realz = n_total_realz
        self.v_log_edges, self.theta_edges = mapping_input.v_log_edges, mapping_input.theta_edges
        self.mapping = mapping
        self.init_class_count = self.get_init_class_count(mapping_input)

    def get_init_class_count(self, map_input):
        index_2d = self.mapping.class_index_2d_vtheta(map_input.initial_v, map_input.initial_theta)
        init_class_count = np.zeros(self.mapping.n_2d_classes)
        for i in index_2d:
            init_class_count[i] += 1
        return init_class_count

    def get_trans_matrix(self, lag, print_every = 50, verbose=True):
        n_2d_class = self.mapping.n_2d_classes
        i_list = []
        j_list = []
        ij_list = set([])
        val_list = []
        for j in range(self.n_total_realz):
            start_idx = 0
            # load the polar coordinates file
            data_path = os.path.join(self.input_folder, 'polar_'+str(j)+'.npz')
            data = np.load(data_path)
            big_v, big_theta, ptr_list = data['V'], data['Theta'], data['ptr']
            for i in ptr_list:
                new_v, new_theta = big_v[start_idx:i], big_theta[start_idx:i]
                start_idx = i
                if len(new_v) > lag:
                    class_2d = self.mapping.class_index_2d_vtheta(new_v, new_theta)
                    fill_one_trajectory_sparse_cython(lag, class_2d, i_list, j_list, ij_list, val_list)
        return csc_matrix((val_list, (i_list, j_list)), shape = (n_2d_class, n_2d_class))