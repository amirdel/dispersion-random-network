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

from py_dp.dispersion.binning import abs_vel_log_bins_low_high, make_theta_bins_linear, \
    make_input_for_binning_v_theta_freq, binning_input_v_theta_freq_y, make_y_bins_linear
import numpy as np
import os as os
import itertools as itertools

class TemporalMapInput(object):
    def __init__(self, input_folder, n_binning_realz, n_absv_class, n_theta_class, time_step,
                 n_slow_class, max_allowed, verbose=True, print_every=40, average_available=False):
        """
        Generate the input for the mapping object for a temporal Markov model, it contains sample arrays for velocity,
        angle and frequency, and the bins for the velocity and angle processes
        :param input_folder:
        :param n_binning_realz:
        :param n_absv_class:
        :param n_theta_class:
        :param time_step:
        :param n_slow_class:
        :param max_allowed:
        """
        self.time_step = time_step
        self.input_folder = input_folder
        self.n_binning_realz = n_binning_realz
        self.n_absv_class = n_absv_class
        self.n_theta_class = n_theta_class
        self.n_slow_class = n_slow_class
        self.max_allowed = max_allowed
        #making sample data for creating bins
        self.v_array, self.theta_array, self.freq_array, self.pointer_list, self.initial_v, self.initial_f, self.initial_theta = \
        self.binning_data(average_available, input_folder, n_binning_realz, time_step, verbose, print_every)
        self.v_log_edges = abs_vel_log_bins_low_high(self.v_array, n_absv_class, n_slow_class, max_allowed)
        self.theta_edges = make_theta_bins_linear(n_theta_class)

    def binning_data(self, average_available, input_folder, n_binning_realz, time_step, verbose, print_every):
        if average_available:
            init_data = np.load(os.path.join(input_folder, 'initial_arrays.npz'))
            initial_v, initial_theta, initial_f = init_data['v'], init_data['theta'],init_data['f']
            big_v, big_theta, big_f, big_ptr = [[] for i in range(4)]
            realz_used = 0
            for file_idx in range(n_binning_realz):
                # load a polar file
                data = np.load(os.path.join(input_folder, 'avg_polar_'+str(file_idx)+'.npz'))
                file_idx += 1
                realz_used += data['n_realz']
                big_v.append(data['V'])
                big_theta.append(data['Theta'])
                big_f.append(data['F'])
                big_ptr.append(data['ptr'])
            #flatten the nested lists
            big_v, big_theta, big_f, big_ptr = [flatten(ll) for ll in [big_v, big_theta, big_f, big_ptr]]
            return big_v, big_theta, big_f, big_ptr, initial_v, initial_f, initial_theta
        else:
            return make_input_for_binning_v_theta_freq(input_folder, n_binning_realz, time_step, verbose=verbose, print_every=print_every)

def flatten(ll):
    chain = itertools.chain(*ll)
    return np.array(list(chain), dtype=np.float)

class SpatialMapInput(object):
    def __init__(self, input_folder, n_binning_realz, n_absv_class, n_slow_class, max_allowed,
                 init_percentile=None):
        self.input_folder = input_folder
        self.n_binning_realz = n_binning_realz
        self.n_absv_class = n_absv_class
        self.n_slow_class = n_slow_class
        self.max_allowed = max_allowed
        #making sample data for creating bins
        self.v_array, self.theta_array, self.pointer_list, self.initial_v, self.initial_theta = \
        self.binning_data(input_folder, n_binning_realz)
        self.v_log_edges = abs_vel_log_bins_low_high(self.v_array, n_absv_class, n_slow_class, max_allowed,
                                                     init_percentile)
        self.theta_edges = make_theta_bins_linear(4)

    def binning_data(self, input_folder, n_binning_realz):
        init_data = np.load(os.path.join(input_folder, 'initial_arrays.npz'))
        initial_v, initial_theta = init_data['v'], init_data['theta']
        big_v, big_theta, big_ptr = [[] for i in range(3)]
        realz_used = 0
        for file_idx in range(n_binning_realz):
            # load a polar file
            data = np.load(os.path.join(input_folder, 'polar_'+str(file_idx)+'.npz'))
            file_idx += 1
            realz_used += data['n_realz']
            big_v.append(data['V'])
            big_theta.append(data['Theta'])
            big_ptr.append(data['ptr'])
        #flatten the nested lists
        big_v, big_theta, big_ptr = [flatten(ll) for ll in [big_v, big_theta, big_ptr]]
        return big_v, big_theta, big_ptr, initial_v, initial_theta

class TemporalMapInputWithY(object):
    def __init__(self, input_folder, n_binning_realz, n_absv_class, n_theta_class, n_y_class, time_step,
                 n_slow_class, max_allowed, verbose=True, print_every=40, average_available=False):
        """
        Generate the input for the mapping object for a temporal Markov model, it contains sample arrays for velocity,
        angle and frequency, and the bins for the velocity and angle processes
        :param input_folder:
        :param n_binning_realz:
        :param n_absv_class:
        :param n_theta_class:
        :param time_step:
        :param n_slow_class:
        :param max_allowed:
        """
        self.time_step = time_step
        self.input_folder = input_folder
        self.n_binning_realz = n_binning_realz
        self.n_absv_class = n_absv_class
        self.n_theta_class = n_theta_class
        self.n_slow_class = n_slow_class
        self.max_allowed = max_allowed
        #making sample data for creating bins
        self.v_array, self.theta_array, self.big_y_array, self.freq_array, self.pointer_list, self.initial_v, \
        self.initial_f, self.initial_theta = \
        self.binning_data(average_available, input_folder, n_binning_realz, time_step, verbose, print_every)
        self.v_log_edges = abs_vel_log_bins_low_high(self.v_array, n_absv_class, n_slow_class, max_allowed)
        self.theta_edges = make_theta_bins_linear(n_theta_class)
        self.y_edges = make_y_bins_linear(self.big_y_array, n_y_class)

    def binning_data(self, average_available, input_folder, n_binning_realz, time_step, verbose, print_every):
        if average_available:
            init_data = np.load(os.path.join(input_folder, 'initial_arrays.npz'))
            initial_v, initial_theta, initial_f = init_data['v'], init_data['theta'],init_data['f']
            big_v, big_theta, big_f, big_ptr, big_y = [[] for i in range(5)]
            realz_used = 0
            for file_idx in range(n_binning_realz):
                # load a polar file
                data = np.load(os.path.join(input_folder, 'avg_polar_'+str(file_idx)+'.npz'))
                realz_used += data['n_realz']
                big_v.append(data['V'])
                big_theta.append(data['Theta'])
                big_f.append(data['F'])
                big_ptr.append(data['ptr'])
                cartesian = np.load(os.path.join(input_folder, 'avg_cartesian_'+str(file_idx)+'.npz'))
                big_y.append(cartesian['Y'])
            #flatten the nested lists
            big_v, big_theta, big_f, big_ptr, big_y = [flatten(ll) for ll in
                                                       [big_v, big_theta, big_f, big_ptr, big_y]]
            return big_v, big_theta, big_y, big_f, big_ptr, initial_v, initial_f, initial_theta
        else:
            return binning_input_v_theta_freq_y(input_folder, n_binning_realz, time_step, verbose=verbose)