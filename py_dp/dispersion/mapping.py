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

from py_dp.dispersion.binning import fix_bisect_left_indices
import random
import numpy as np
import bisect as bs
from py_dp.dispersion.transition_matrix_fcns import fix_out_of_bound
from copy import copy
from py_dp.dispersion.convert_to_time_process_with_freq import remove_duplicate_xy

class mapping_v_sgn_repeat(object):
    """
    A class for two way mapping between velocity, direction of velocity and the number of repeats for that velocity and
    a 3d class number. Used for 1d results.
    """
    def __init__(self, v_log_edges, cumsum_n_subclass, sub_classes_nrepeat):
        self.v_log_edges = v_log_edges
        self.cumsum_n_subclass = cumsum_n_subclass
        self.sub_classes_nrepeat = sub_classes_nrepeat
        self.n_abs_v_classes = len(v_log_edges) - 1
        self.n_2d_classes = 2*self.n_abs_v_classes
        self.n_3d_classes = cumsum_n_subclass[-1]

    def draw_from_class_velocity(self, idx):
        v_log_edges = self.v_log_edges
        x = random.random()
        log_v = v_log_edges[idx]*x + v_log_edges[idx+1]*(1-x)
        return np.exp(log_v)


    def find_3d_class_number(self, index_2d, freq_array):
        #find the (v, sgn(v)) class number
        #index_2d = class_index_abs_log(v_array, v_log_edges)
        sub_classes_nrepeat = self.sub_classes_nrepeat
        cumsum_n_subclass = self.cumsum_n_subclass
        ind_3d_array = np.zeros(len(index_2d))
        for i in range(len(index_2d)):
            class_2d = index_2d[i]
            freq = freq_array[i]
            sub_class_array = sub_classes_nrepeat[class_2d]
            ind_repeat = np.array(bs.bisect_left(sub_class_array, freq))
            #check out of bounds
            fix_bisect_left_indices(ind_repeat, sub_class_array)
            ind_3d_array[i] = cumsum_n_subclass[class_2d] + ind_repeat
        return ind_3d_array

    def find_v_sgn_freq(self, index_3d):
        """
        :return: abs(v), sgn(v), frequency
        """
        if index_3d>=self.n_3d_classes:
            raise('index of 3d class can not be larger than number of classes - ')
        cumsum_n_subclass = self.cumsum_n_subclass
        last_idx_array = self.cumsum_n_subclass[1:] - 1
        sub_classes_nrepeat = self.sub_classes_nrepeat
        class_2d = bs.bisect_left(last_idx_array, index_3d)
        #check out of bounds
        class_2d -= (class_2d == self.n_2d_classes)
        freq_idx = index_3d - cumsum_n_subclass[class_2d]
        freq_array = sub_classes_nrepeat[class_2d]
        freq = freq_array[freq_idx]
        v_class_1d = (class_2d - np.mod(class_2d,2))/2
        sgn_v = -1.0 + 2.0*(np.mod(class_2d,2) == 0)
        abs_v = self.draw_from_class_velocity(v_class_1d)
        return abs_v, sgn_v, freq

    def find_absvclass_sgn_freq(self, index_3d):
        """
        :return: abs_v_class, sgn(v), frequency
        """
        if index_3d>=self.n_3d_classes:
            raise('index of 3d class can not be larger than number of classes - ')
        cumsum_n_subclass = self.cumsum_n_subclass
        last_idx_array = self.cumsum_n_subclass[1:] - 1
        sub_classes_nrepeat = self.sub_classes_nrepeat
        class_2d = bs.bisect_left(last_idx_array, index_3d)
        #check out of bounds
        class_2d -= (class_2d == self.n_2d_classes)
        freq_idx = index_3d - cumsum_n_subclass[class_2d]
        freq_array = sub_classes_nrepeat[class_2d]
        freq = freq_array[freq_idx]
        v_class_1d = (class_2d - np.mod(class_2d,2))/2
        sgn_v = -1.0 + 2.0*(np.mod(class_2d,2) == 0)
        return v_class_1d, sgn_v, freq

class mapping(object):
    def __init__(self):
        pass

    def find_1d_class_idx(self, input_array, input_bins):
        class_idx = np.digitize(input_array, input_bins)
        fix_out_of_bound(class_idx, input_bins)
        return class_idx - 1

    def find_2d_class_idx(self, input_array_1, input_bins_1, input_array_2, input_bins_2):
        n_second_class = len(input_bins_2) - 1
        first_class = self.find_1d_class_idx(input_array_1, input_bins_1)
        second_class = self.find_1d_class_idx(input_array_2, input_bins_2)
        class_index_2d = first_class * n_second_class + second_class
        return np.array(class_index_2d, dtype=np.int)

    def class_idx_1d_from_2d(self, class_idx_2d, n_second_class):
        second_class_index = np.mod(class_idx_2d, n_second_class)
        first_class_index = np.divide(class_idx_2d - second_class_index, n_second_class)
        return first_class_index, second_class_index


class mapping_v_theta_repeat(mapping):
    """
    class to map between log of absolute value of velocity and angle and frequency to 2d and 3d bins, used for 2d case
    """

    def __init__(self, v_log_edges, theta_edges, v_array, theta_array, f_array, make_aux_arrays=True):
        """
        :param v_log_edges: edges for log(abs(v))
        :param theta_edges: edges for theta bins
        :param v_array: input v array to build mapping
        :param theta_array: input theta array to build mapping
        :param f_array: input f array to build mapping
        """
        self.v_log_edges = v_log_edges
        self.theta_edges = theta_edges
        self.n_abs_v_classes = len(v_log_edges) - 1
        self.n_theta_classes = len(theta_edges) - 1
        self.n_2d_classes = self.n_abs_v_classes * self.n_theta_classes
        if make_aux_arrays:
            #we only make the auxilary arrays for cases with 3d classes
            self.sub_classes_nrepeat, self.cumsum_n_subclass = self.make_auxillary_arrays(v_array, theta_array, f_array)
            self.n_3d_classes = self.cumsum_n_subclass[-1]

    def make_auxillary_arrays(self, v_array, theta_array, f_array):
        print 'making auxillary arrays for mapping...'
        n_2d_classes = self.n_2d_classes
        vtheta_class_number = self.class_index_2d_vtheta(v_array, theta_array)
        sub_classes_nrepeat = []
        n_subclass = []
        place_holder = np.array([1], dtype=np.int)
        # loop over v-theta classes
        for i in range(n_2d_classes):
            possible_f_vals = np.unique(f_array[vtheta_class_number == i])
            if not len(possible_f_vals):
                possible_f_vals = copy(place_holder)
            sub_classes_nrepeat.append(sorted(possible_f_vals))
            n_subclass.append(len(possible_f_vals))
        modified_n_sub_class = np.array(n_subclass)
        cumsum_n_subclass = np.hstack((0, np.cumsum(modified_n_sub_class)))
        print 'done'
        return sub_classes_nrepeat, cumsum_n_subclass

    def class_index_2d_vtheta(self, v_array, theta_array):
        """
        theta_values between 0 and 2pi
        starting from first v class v0t0:0, v0t1:1, ..., v0tn:n-1, ... , vn-1tn-1:
        """
        return self.find_2d_class_idx(np.log(np.abs(v_array)), self.v_log_edges, theta_array, self.theta_edges)

    def class_index_1d_v_theta_from_2d(self, class_array_2d):
        """
        Take a 2d v_theta class and return the indices of 1d v and theta classes
        :param class_array_2d:
        :return: v_class_index, theta_class_index
        """
        n_theta_classes = self.n_theta_classes
        v_class_index, theta_class_index = self.class_idx_1d_from_2d(class_array_2d, n_theta_classes)
        return v_class_index, theta_class_index

    def draw_from_class_velocity(self, idx):
        v_log_edges = self.v_log_edges
        x = random.random()
        log_v = v_log_edges[idx] * x + v_log_edges[idx + 1] * (1 - x)
        return np.exp(log_v)

    def draw_from_class_theta(self, idx):
        x = random.random()
        theta_edges = self.theta_edges
        theta = theta_edges[idx] * x + theta_edges[idx + 1] * (1 - x)
        return theta

    def find_3d_class_number(self, index_2d, freq_array):
        assert(len(index_2d) == len(freq_array))
        sub_classes_nrepeat = self.sub_classes_nrepeat
        cumsum_n_subclass = self.cumsum_n_subclass
        ind_3d_array = np.zeros(len(index_2d), dtype=np.int)
        for i in range(len(index_2d)):
            class_2d = index_2d[i]
            freq = freq_array[i]
            sub_class_array = sub_classes_nrepeat[class_2d]
            ind_repeat = np.array(bs.bisect_left(sub_class_array, freq))
            # check out of bounds
            fix_bisect_left_indices(ind_repeat, sub_class_array)
            ind_3d_array[i] = cumsum_n_subclass[class_2d] + ind_repeat
        return ind_3d_array


    def find_v_theta_freq(self, index_3d):
        """
        :return: abs(v), theta, frequency
        """
        if np.any(index_3d >= self.n_3d_classes):
            raise ('index of 3d class can not be larger than number of classes - ')
        index_3d = int(index_3d)
        cumsum_n_subclass = self.cumsum_n_subclass
        last_idx_array = self.cumsum_n_subclass[1:] - 1
        sub_classes_nrepeat = self.sub_classes_nrepeat
        class_2d = bs.bisect_left(last_idx_array, index_3d)
        # check out of bounds
        class_2d -= (class_2d == self.n_2d_classes)
        freq_idx = index_3d - cumsum_n_subclass[class_2d]
        freq_array = sub_classes_nrepeat[class_2d]
        freq = freq_array[freq_idx]
        v_idx, theta_idx = self.class_index_1d_v_theta_from_2d(class_2d)
        abs_v = self.draw_from_class_velocity(v_idx)
        theta = self.draw_from_class_theta(theta_idx)
        return abs_v, theta, freq

class mapping_v_theta_y(mapping):
    """
    class to map between log of absolute value of velocity and angle and frequency to 2d and 3d bins, used for 2d case
    """

    def __init__(self, v_log_edges, theta_edges, y_edges):
        """
        :param v_log_edges: edges for log(abs(v))
        :param theta_edges: edges for theta bins
        :param y_edges: edges for y bins
        """
        self.v_log_edges = v_log_edges
        self.theta_edges = theta_edges
        self.y_edges = y_edges
        self.n_abs_v_classes = len(v_log_edges) - 1
        self.n_theta_classes = len(theta_edges) - 1
        self.n_y_classes = len(y_edges) - 1
        self.n_2d_classes = self.n_y_classes * self.n_theta_classes

    def class_index_2d_theta_y(self, theta_array, y_array):
        """
        theta_values between 0 and 2pi
        starting from first v class v0t0:0, v0t1:1, ..., v0tn:n-1, ... , vn-1tn-1:
        """
        return self.find_2d_class_idx(theta_array, self.theta_edges, y_array, self.y_edges)

    def class_index_1d_theta_y_from_2d(self, class_array_2d):
        """
        Take a 2d v_theta class and return the indices of 1d v and theta classes
        :param class_array_2d:
        :return: v_class_index, theta_class_index
        """
        n_y_classes = self.n_y_classes
        theta_class_index, y_class_index = self.class_idx_1d_from_2d(class_array_2d, n_y_classes)
        return theta_class_index, y_class_index