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

import random as random
import bisect as bs
import numpy as np
from py_dp.dispersion.binning import get_cdf
from py_dp.dispersion.trajectory_count_cython import choose_next_class_vector_cython

class dispersionModelGeneral(object):
    """
    mother class for all dispersion models
    """
    def __init__(self, n_particles, n_steps, inj_location = "start", verbose = True):
        self.verbose = verbose
        self.n_particles = n_particles
        self.n_steps = n_steps
        self.inj_location = inj_location
        self.time_array = np.zeros((n_particles,n_steps+1))
        self.x_array = np.zeros((n_particles,n_steps+1))
        self.y_array = np.zeros((n_particles, n_steps + 1))
        self.last_index_array = n_steps*np.ones(n_particles, dtype=np.int8)


class DispModelJoint(dispersionModelGeneral):
    def __init__(self, n_particles, n_steps, dt, x_max, trans_matrix,
                 mapping, init_class_count, inj_location = "start",
                 verbose = True):
        super(DispModelJoint, self).__init__(n_particles, n_steps, inj_location, verbose)
        self.trans_matrix = trans_matrix
        self.init_class_count = init_class_count
        self.mapping = mapping
        self.dt = dt
        self.x_max = x_max
        self.init_class_cdf = get_cdf(init_class_count)
        self.n_class = np.sqrt(trans_matrix.shape[0])
        self.blocked_particles = []
        self.fix_matrix_types()

    def choose_next_class_vector(self, current_class):
        return choose_next_class_vector_cython(self.trans_matrix.indptr,
                                               self.trans_matrix.indices,
                                               self.trans_matrix.data, current_class)

    def draw_from_class_velocity(self, idx):
        v_log_edges = self.mapping.v_log_edges
        x = np.random.rand(len(idx))
        log_v = np.multiply(v_log_edges[idx],x) + np.multiply(v_log_edges[idx + 1] ,1 - x)
        return np.exp(log_v)

    def draw_from_class_theta(self, idx):
        x = np.random.rand(len(idx))
        theta_edges = self.mapping.theta_edges
        theta = np.multiply(theta_edges[idx], x) + np.multiply(theta_edges[idx + 1], 1 - x)
        return theta

    def fix_matrix_types(self):
        self.trans_matrix.indptr = np.array(self.trans_matrix.indptr, dtype=np.int)
        self.trans_matrix.indices = np.array(self.trans_matrix.indices, dtype=np.int)
        self.trans_matrix.data = np.array(self.trans_matrix.data, dtype=np.float)

class DispModelExtendedStencil(DispModelJoint):
    def __init__(self, n_particles, n_steps, dt, x_max, trans_matrix,
                 mapping, init_class_count, inj_location = "start",
                 verbose = True):
        super(DispModelExtendedStencil, self).__init__(n_particles, n_steps, dt, x_max, trans_matrix,
                                                       mapping, init_class_count, inj_location, verbose)

    def find_v_theta_freq_vector(self, class_3d_vec):
        cumsum_n_subclass = self.mapping.cumsum_n_subclass
        last_idx_array = self.mapping.cumsum_n_subclass[1:] - 1
        sub_classes_nrepeat = self.mapping.sub_classes_nrepeat
        class_2d = np.searchsorted(last_idx_array, class_3d_vec)
        # check out of bounds
        class_2d -= (class_2d == self.mapping.n_2d_classes)
        freq_idx_array = class_3d_vec - cumsum_n_subclass[class_2d]
        freq = np.zeros(len(class_3d_vec), dtype=np.int)
        for i, idx_2d in enumerate(class_2d):
            freq_array = sub_classes_nrepeat[idx_2d]
            freq[i] = freq_array[freq_idx_array[i]]
        # TODO maybe: replace with cython function
        # freq_array = find_freq_cython(sub_classes_nrepeat, class_2d, freq_idx)
        v_idx, theta_idx = self.mapping.class_index_1d_v_theta_from_2d(class_2d)
        abs_v = self.draw_from_class_velocity(v_idx)
        theta = self.draw_from_class_theta(theta_idx)
        return abs_v, theta, freq

    def follow_all_particles_vector(self, verbose=True, print_every=50):
        dt = self.dt
        x_array = self.x_array
        y_array = self.y_array
        t_array = self.time_array
        n_particles = self.n_particles
        # class_idx_vec = np.array(np.searchsorted(self.init_class_cdf, np.random.rand(n_particles))-1, dtype=np.int)
        class_idx_vec = np.array(np.searchsorted(self.init_class_cdf, np.random.rand(n_particles)), dtype=np.int)
        keep_mask = np.ones(n_particles, dtype=bool)
        #loop over all steps and save dx, dy, dt
        for i in range(self.n_steps):
            if verbose and not i%print_every:
                print 'step number: ',i
            v_array, theta_array, f_array = self.find_v_theta_freq_vector(class_idx_vec)
            x_array[:, i+1] = np.multiply(np.multiply(v_array, np.cos(theta_array)), dt*f_array)
            y_array[:, i+1] = np.multiply(np.multiply(v_array, np.sin(theta_array)), dt*f_array)
            t_array[:, i+1] = dt*f_array
            next_idx_vec = self.choose_next_class_vector(class_idx_vec)
            keep_mask[next_idx_vec==-12] = False
            class_idx_vec = next_idx_vec
            class_idx_vec[~keep_mask] = 0
        x_array = x_array[keep_mask, :]
        self.x_array = np.cumsum(x_array, axis=1)
        y_array = y_array[keep_mask, :]
        self.y_array = np.cumsum(y_array, axis=1)
        t_array = t_array[keep_mask, :]
        self.time_array = np.cumsum(t_array, axis=1)


class DispModelStencilMethod(DispModelJoint):
    def __init__(self, n_particles, n_steps, dt, x_max, trans_matrix,
                 mapping, init_class_count, inj_location = "start",
                 verbose = True):
        super(DispModelStencilMethod, self).__init__(n_particles, n_steps, dt, x_max, trans_matrix,
                 mapping, init_class_count, inj_location, verbose)

    def follow_all_particles_vector(self, verbose=True, print_every=50):
        self.trans_matrix.indptr = np.array(self.trans_matrix.indptr ,dtype=np.int)
        self.trans_matrix.indices = np.array(self.trans_matrix.indices, dtype=np.int)
        self.trans_matrix.data = np.array(self.trans_matrix.data, dtype=np.float)
        dt = self.dt
        x_array = self.x_array
        y_array = self.y_array
        t_array = self.time_array
        n_particles = self.n_particles
        # class_idx_vec = np.array(np.searchsorted(self.init_class_cdf, np.random.rand(n_particles))-1, dtype=np.int)
        class_idx_vec = np.array(np.searchsorted(self.init_class_cdf, np.random.rand(n_particles)), dtype=np.int)
        keep_mask = np.ones(n_particles, dtype=bool)
        #loop over all steps and save dx, dy, dt
        for i in range(self.n_steps):
            if verbose and not i%print_every:
                print 'step number: ',i
            v_idx, theta_idx = self.mapping.class_index_1d_v_theta_from_2d(class_idx_vec)
            v_array = self.draw_from_class_velocity(v_idx)
            theta_array = self.draw_from_class_theta(theta_idx)
            x_array[:, i+1] = np.multiply(v_array, np.cos(theta_array))*dt
            y_array[:, i+1] = np.multiply(v_array, np.sin(theta_array))*dt
            t_array[:, i+1] = dt
            next_idx_vec = self.choose_next_class_vector(class_idx_vec)
            keep_mask[next_idx_vec==-12] = False
            class_idx_vec = next_idx_vec
            class_idx_vec[~keep_mask] = 0
        x_array = x_array[keep_mask, :]
        self.x_array = np.cumsum(x_array, axis=1)
        y_array = y_array[keep_mask, :]
        self.y_array = np.cumsum(y_array, axis=1)
        t_array = t_array[keep_mask, :]
        self.time_array = np.cumsum(t_array, axis=1)

class DispModelSpace(DispModelJoint):
    def __init__(self, l, n_particles, n_steps, x_max, trans_matrix,
                 mapping, init_class_count, inj_location="start",
                 verbose=True):
        super(DispModelSpace, self).__init__(n_particles, n_steps, None, x_max, trans_matrix,
                                             mapping, init_class_count, inj_location, verbose)
        self.l = l

    def draw_from_class_theta(self, idx):
        theta_centers = np.pi*np.array([-0.75, -0.25, 0.25, 0.75])
        return theta_centers[idx]

    def follow_all_particles_vector(self, verbose=True, print_every=50):
        l = self.l
        x_array = self.x_array
        y_array = self.y_array
        t_array = self.time_array
        n_particles = self.n_particles
        class_idx_vec = np.array(np.searchsorted(self.init_class_cdf, np.random.rand(n_particles)), dtype=np.int)
        keep_mask = np.ones(n_particles, dtype=bool)
        # loop over all steps and save dx, dy, dt
        for i in range(self.n_steps):
            if verbose and not i % print_every:
                print 'step number: ', i
            v_idx, theta_idx = self.mapping.class_index_1d_v_theta_from_2d(class_idx_vec)
            v_array = self.draw_from_class_velocity(v_idx)
            theta_array = self.draw_from_class_theta(theta_idx)
            dt_array = l/v_array
            x_array[:, i + 1] = x_array[:,i] + np.multiply(v_array*np.cos(theta_array), dt_array)
            y_array[:, i + 1] = y_array[:,i] + np.multiply(v_array*np.sin(theta_array), dt_array)
            t_array[:, i + 1] = t_array[:,i] + dt_array
            next_idx_vec = self.choose_next_class_vector(class_idx_vec)
            keep_mask[next_idx_vec == -12] = False
            class_idx_vec = next_idx_vec
            class_idx_vec[~keep_mask] = 0
        x_array = x_array[keep_mask, :]
        y_array = y_array[keep_mask, :]
        t_array = t_array[keep_mask, :]


class DispModelUncorrelated(dispersionModelGeneral):
    def __init__(self, n_particles, n_steps, dt, sample_v, sample_theta, sample_f,
                 init_v, init_theta, init_f, inj_location="start", verbose=True):
        super(DispModelUncorrelated, self).__init__(n_particles, n_steps, inj_location, verbose)
        self.init_v = init_v
        self.init_theta = init_theta
        self.init_f = init_f
        self.dt = dt
        self.v_dist = sample_v
        self.theta_dist = sample_theta
        self.f_dist = sample_f

    def follow_all_particles_vector(self, verbose=True, print_every=50):
        x_array = self.x_array
        y_array = self.y_array
        t_array = self.time_array
        n_particles = self.n_particles
        len_init = len(self.init_v)
        # draw random init idx and initialize v, theta, f
        init_idx = np.random.randint(0, len_init, n_particles)
        v_array = self.init_v[init_idx]
        theta_array = self.init_theta[init_idx]
        f_array = self.init_f[init_idx]
        dt = self.dt
        v_dist, theta_dist, f_dist = self.v_dist, self.theta_dist, self.f_dist
        len_dist = len(v_dist)
        for i in range(self.n_steps):
            if verbose and not i % print_every:
                print 'step number: ', i
            x_array[:, i+1] = x_array[:,i] + np.multiply(np.multiply(v_array, np.cos(theta_array)), dt*f_array)
            y_array[:, i+1] = y_array[:,i] + np.multiply(np.multiply(v_array, np.sin(theta_array)), dt*f_array)
            t_array[:, i+1] = t_array[:,i] + dt*f_array
            next_idx_vec = np.random.randint(0, len_dist, n_particles)
            v_array = v_dist[next_idx_vec]
            theta_array = theta_dist[next_idx_vec]
            f_array = f_dist[next_idx_vec]