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
from py_dp.dispersion.second_order_markov import find_1d_bins
from py_dp.dispersion.dispersion_models import dispersionModelGeneral

# These classes are not used now. Just kept for reference.

class dispModelUncorrStencil(dispersionModelGeneral):
    def __init__(self, n_particles, n_steps, dt, dx_array, x_max,
                 inj_location = "start", verbose = True):
        super(dispModelUncorrStencil,self).__init__(n_particles, n_steps,
                                                    inj_location, verbose)
        self.dx_array = dx_array
        self.dt = dt
        self.table_size = len(dx_array) - 1
        self.x_max = x_max

    def advance_one_step(self, particle_number, current_index):
        x_max = self.x_max
        end_of_domain_reached = False
        dx_array = self.dx_array
        dt = self.dt
        table_size = self.table_size
        rand_ind = random.randint(0,table_size)
        dx = dx_array[rand_ind]
        current_t = self.time_array[particle_number, current_index]
        current_x = self.x_array[particle_number, current_index]
        next_x = current_x + dx
        if next_x > x_max:
            velocity = dx/dt
            distance_to_end = x_max - current_x
            dt = distance_to_end/velocity
            next_x = x_max
            end_of_domain_reached = True
        next_index = current_index + 1
        self.time_array[particle_number,next_index] = current_t + dt
        self.x_array[particle_number,next_index] = next_x
        return end_of_domain_reached

    def follow_specific_paticle(self,particle_number):
        n_steps = self.n_steps
        for step in range(n_steps):
            #x_array, pore_nr_array, time_array entries are changed inside
            #advance_to_next_pore
            current_index = step
            end_flag = self.advance_one_step(particle_number, current_index)
            if end_flag:
                freeze_idx = current_index + 1
                self.freeze_particle(particle_number, freeze_idx)
                break

    def follow_all_particles(self):
        n_particles = self.n_particles
        for particle in range(n_particles):
            self.follow_specific_paticle(particle)

    def freeze_particle(self,particle_number,current_index):
        """
        after a particle gets to the end of the domain it would stay there.
        this function would copy the value at current_idx to all following values for x and time
        """
        self.x_array[particle_number,current_index:] = self.x_array[particle_number,current_index]
        self.time_array[particle_number,current_index:] = self.time_array[particle_number,current_index]
        #self.freeze_time[particle_number] = self.time_array[particle_number,current_index]
        self.last_index_array[particle_number] = current_index

class dispModelCorrelatedStencil(dispersionModelGeneral):
    def __init__(self, n_particles, n_steps, dt, x_max, trans_matrix,
                 class_velocity, init_class_count, inj_location = "start",
                 verbose = True):
        super(dispModelCorrelatedStencil,self).__init__(n_particles, n_steps,
                                                        inj_location, verbose)
        self.trans_matrix = trans_matrix
        self.init_class_count = init_class_count
        self.class_velocity = class_velocity
        self.dt = dt
        self.x_max = x_max
        self.init_class_cdf = get_cdf(init_class_count)
        self.cdf_matrix = np.cumsum(trans_matrix, axis=0)

    def draw_from_init_calss_idx(self):
        return bs.bisect(self.init_class_cdf, random.random())

    def choose_next_class(self, current_class):
        cdf = self.cdf_matrix[:, current_class]
        return bs.bisect(cdf, random.random())

    def follow_one_particle(self, particle_number):
        dt = self.dt
        class_velocity = self.class_velocity
        x_array = self.x_array
        t_array = self.time_array
        x = 0.0
        t = 0.0
        out_put_idx = 1
        #initialize the particle velocity
        class_idx = self.draw_from_init_calss_idx()
        next_idx = 0
        v = class_velocity[class_idx]
        idx_max = self.n_steps + 1
        while out_put_idx < idx_max:
            x += dt*v
            t += dt
            x_array[particle_number, out_put_idx] = x
            t_array[particle_number, out_put_idx] = t
            out_put_idx += 1
            next_idx = self.choose_next_class(class_idx)
            v = class_velocity[next_idx]
            class_idx = next_idx

    def follow_all_particles(self):
        for i in range(self.n_particles):
            self.follow_one_particle(i)

class dispModelCorrelatedStencilFix(dispModelCorrelatedStencil):
    def __init__(self, n_particles, n_steps, dt, x_max, trans_matrix,
                 class_velocity, init_class_count, length, inj_location = "start", verbose = True):
        super(dispModelCorrelatedStencilFix,self).__init__(n_particles, n_steps, dt, x_max, trans_matrix,
                 class_velocity, init_class_count, inj_location, verbose)
        self.length = length

    def follow_one_particle(self, particle_number):
        l = self.length
        dt = self.dt
        class_velocity = self.class_velocity
        x_array = self.x_array
        t_array = self.time_array
        x = 0.0
        t = 0.0
        out_put_idx = 1
        # initialize the particle velocity
        class_idx = self.draw_from_init_calss_idx()
        next_idx = 0
        v = class_velocity[class_idx]
        idx_max = self.n_steps + 1
        while out_put_idx < idx_max:
            dx = v*dt
            abs_dx = abs(dx)
            if abs_dx < l:
                length_traveled = 0.0
                while abs(length_traveled) <= l - abs_dx and out_put_idx < idx_max:
                    length_traveled += dx
                    x += dx
                    t += dt
                    x_array[particle_number, out_put_idx] = x
                    t_array[particle_number, out_put_idx] = t
                    out_put_idx += 1
            else:
                x += dt * v
                t += dt
                x_array[particle_number, out_put_idx] = x
                t_array[particle_number, out_put_idx] = t
                out_put_idx += 1
            next_idx = self.choose_next_class(class_idx)
            v = class_velocity[next_idx]
            class_idx = next_idx

class dispModelCorrelatedSpace(dispersionModelGeneral):
    def __init__(self, n_particles, n_steps, dx, x_max, trans_matrix,
                 class_velocity, init_class_count, inj_location = "start",
                 verbose = True):
        super(dispModelCorrelatedSpace,self).__init__(n_particles, n_steps,
                                                        inj_location, verbose)
        self.trans_matrix = trans_matrix
        self.init_class_count = init_class_count
        self.class_velocity = class_velocity
        self.dx = dx
        self.x_max = x_max
        self.init_class_cdf = get_cdf(init_class_count)
        self.cdf_matrix = np.cumsum(trans_matrix, axis=0)

    def draw_from_init_calss_idx(self):
        return bs.bisect(self.init_class_cdf, random.random())

    def choose_next_class(self, current_class):
        cdf = self.cdf_matrix[:, current_class]
        return bs.bisect(cdf, random.random())

    def follow_one_particle(self, particle_number):
        dx = self.dx
        class_velocity = self.class_velocity
        x_array = self.x_array
        t_array = self.time_array
        x = 0.0
        t = 0.0
        out_put_idx = 1
        #initialize the particle velocity
        class_idx = self.draw_from_init_calss_idx()
        v = class_velocity[class_idx]
        idx_max = self.n_steps + 1
        while out_put_idx < idx_max:
            x += np.sign(v)*dx
            t += dx/abs(v)
            x_array[particle_number, out_put_idx] = x
            t_array[particle_number, out_put_idx] = t
            out_put_idx += 1
            next_idx = self.choose_next_class(class_idx)
            v = class_velocity[next_idx]
            class_idx = next_idx

    def follow_all_particles(self):
        for i in range(self.n_particles):
            self.follow_one_particle(i)

class dispModelCorrelatedSpaceKang(dispersionModelGeneral):
    def __init__(self, n_particles, n_steps, dx, x_max, trans_matrix,
                 class_log_edges, init_class_count, inj_location = "start",
                 verbose = True):
        super(dispModelCorrelatedSpaceKang,self).__init__(n_particles, n_steps,
                                                        inj_location, verbose)
        self.trans_matrix = trans_matrix
        self.init_class_count = init_class_count
        self.class_log_edges = class_log_edges
        self.class_velocity = self.get_class_velocity(class_log_edges)
        self.dx = dx
        self.x_max = x_max
        self.init_class_cdf = get_cdf(init_class_count)
        self.cdf_matrix = np.cumsum(trans_matrix, axis=0)

    def get_class_velocity(self, class_log_edges):
        v_log_edges = self.class_log_edges
        n_class = len(class_log_edges) - 1
        class_velocity = np.zeros(n_class)
        for i in range(n_class):
            log_value = 0.5*(v_log_edges[i] + v_log_edges[i+1])
            class_velocity[i] = np.exp(log_value)
        return class_velocity

    def draw_from_class_velocity(self, idx):
        v_log_edges = self.class_log_edges
        x = random.random()
        log_v = v_log_edges[idx]*x + v_log_edges[idx+1]*(1-x)
        return np.exp(log_v)

    def draw_from_init_calss_idx(self):
        return bs.bisect(self.init_class_cdf, random.random())

    def choose_next_class(self, current_class):
        cdf = self.cdf_matrix[:, current_class]
        return bs.bisect(cdf, random.random())

    def follow_one_particle(self, particle_number):
        dx = self.dx
        class_velocity = self.class_velocity
        x_array = self.x_array
        t_array = self.time_array
        x = 0.0
        t = 0.0
        out_put_idx = 1
        #initialize the particle velocity
        v_class_idx = self.draw_from_init_calss_idx()
        class_idx = 2*v_class_idx
        v = self.draw_from_class_velocity(v_class_idx)
        v_sign = 1.0
        idx_max = self.n_steps + 1
        while out_put_idx < idx_max:
            x += v_sign*dx
            t += dx/v
            x_array[particle_number, out_put_idx] = x
            t_array[particle_number, out_put_idx] = t
            out_put_idx += 1
            next_idx = self.choose_next_class(class_idx)
            v_class_idx = np.floor(next_idx/2)
            v_sign = -1.0 + 2.0*((next_idx - 2*v_class_idx) == 0)
            v = self.draw_from_class_velocity(v_class_idx)
            class_idx = next_idx

    def follow_all_particles(self):
        for i in range(self.n_particles):
            self.follow_one_particle(i)

class dispModelCorrelatedStencilKang(dispersionModelGeneral):
    """
    Class to model plume spreading using a Markov model in time, The velocity is
    binned using the binning strategy in Kang 2010
    """
    def __init__(self, n_particles, n_steps, dt, x_max, trans_matrix,
                 class_log_edges, init_class_count, inj_location = "start",
                 verbose = True):
        super(dispModelCorrelatedStencilKang,self).__init__(n_particles, n_steps,
                                                        inj_location, verbose)
        self.trans_matrix = trans_matrix
        self.init_class_count = init_class_count
        self.class_log_edges = class_log_edges
        self.dt = dt
        self.x_max = x_max
        self.init_class_cdf = get_cdf(init_class_count)
        self.cdf_matrix = np.cumsum(trans_matrix, axis=0)

    def draw_from_init_calss_idx(self):
        return bs.bisect(self.init_class_cdf, random.random())

    def choose_next_class(self, current_class):
        cdf = self.cdf_matrix[:, current_class]
        return bs.bisect(cdf, random.random())

    def draw_from_class_velocity(self, idx):
        v_log_edges = self.class_log_edges
        x = random.random()
        log_v = v_log_edges[idx]*x + v_log_edges[idx+1]*(1-x)
        return np.exp(log_v)

    def follow_one_particle(self, particle_number):
        dt = self.dt
        x_array = self.x_array
        t_array = self.time_array
        x = 0.0
        t = 0.0
        out_put_idx = 1
        #initialize the particle velocity
        v_class_idx = self.draw_from_init_calss_idx()
        class_idx = 2*v_class_idx
        #v is the abs value of velocity
        v = self.draw_from_class_velocity(v_class_idx)
        v_sign = 1.0
        idx_max = self.n_steps + 1
        while out_put_idx < idx_max:
            x += dt*v*v_sign
            t += dt
            x_array[particle_number, out_put_idx] = x
            t_array[particle_number, out_put_idx] = t
            out_put_idx += 1
            next_idx = self.choose_next_class(class_idx)
            v_class_idx = np.floor(next_idx/2)
            v_sign = -1.0 + 2.0*((next_idx - 2*v_class_idx) == 0)
            v = self.draw_from_class_velocity(v_class_idx)
            class_idx = next_idx

    def follow_all_particles(self):
        for i in range(self.n_particles):
            self.follow_one_particle(i)

class dispModelOrderTwo(dispersionModelGeneral):
    def __init__(self, n_particles, n_steps, dx, x_max, trans_matrix,
                 class_log_edges, init_class_count, inj_location = "start",
                 verbose = True):
        super(dispModelOrderTwo,self).__init__(n_particles, n_steps,
                                                        inj_location, verbose)
        self.trans_matrix = trans_matrix
        self.init_class_count = init_class_count
        self.class_log_edges = class_log_edges
        self.class_velocity = self.get_class_velocity(class_log_edges)
        self.dx = dx
        self.x_max = x_max
        self.init_class_cdf = get_cdf(init_class_count)
        self.n_class = np.sqrt(trans_matrix.shape[0])
        self.blocked_particles = []

    def get_class_velocity(self, class_log_edges):
        v_log_edges = self.class_log_edges
        n_class = len(class_log_edges) - 1
        class_velocity = np.zeros(n_class)
        for i in range(n_class):
            log_value = 0.5*(v_log_edges[i] + v_log_edges[i+1])
            class_velocity[i] = np.exp(log_value)
        return class_velocity

    def draw_from_class_velocity(self, idx):
        v_log_edges = self.class_log_edges
        x = random.random()
        log_v = v_log_edges[idx]*x + v_log_edges[idx+1]*(1-x)
        return np.exp(log_v)

    def draw_from_init_calss_idx(self):
        return bs.bisect_right(self.init_class_cdf, random.random())

    def choose_next_class(self, current_class):
        indptr = self.trans_matrix.indptr
        start = indptr[current_class]
        end = indptr[current_class+1]
        rows = self.trans_matrix.indices[start:end]
        values = self.trans_matrix.data[start:end]
        if len(values) == 0:
            return -12
        cdf = get_cdf(values)
        return rows[bs.bisect(cdf, random.random())]

    def advance_x_t(self, v, v_sign, x, t):
        t2 = t + self.dx/v
        x2 = x + v_sign*self.dx
        return x2, t2

    def follow_one_particle(self, particle_number):
        n_class = self.n_class
        dx = self.dx
        class_velocity = self.class_velocity
        x_array = self.x_array
        t_array = self.time_array
        x = 0.0
        t = 0.0
        out_put_idx = 1
        #initialize the particle velocity
        #class_idx is the index of the 2d class
        class_idx = self.draw_from_init_calss_idx()
        #i, ip1 are indices of (abs(v), sgn(v)) classes
        i, ip1 = find_1d_bins(class_idx, n_class)
        v_class_idx = np.floor(i/2)
        v_sign = -1.0 + 2.0*((i - 2*v_class_idx) == 0)
        v = self.draw_from_class_velocity(v_class_idx)
        x,t = self.advance_x_t(v, v_sign, x, t)
        x_array[particle_number, out_put_idx] = x
        t_array[particle_number, out_put_idx] = t
        out_put_idx += 1
        v_class_idx = np.floor(ip1/2)
        v_sign = -1.0 + 2.0*((ip1 - 2*v_class_idx) == 0)
        v = self.draw_from_class_velocity(v_class_idx)
        idx_max = self.n_steps + 1
        while out_put_idx < idx_max:
            x,t = self.advance_x_t(v, v_sign, x,t)
            x_array[particle_number, out_put_idx] = x
            t_array[particle_number, out_put_idx] = t
            out_put_idx += 1
            next_idx = self.choose_next_class(class_idx)
            if next_idx == -12:
                self.blocked_particles.append(particle_number)
                return
            t1, t2 = find_1d_bins(next_idx, n_class)
            class_idx = next_idx
            i, ip1 = find_1d_bins(class_idx, n_class)
            v_class_idx = np.floor(ip1/2)
            v_sign = -1.0 + 2.0*((ip1 - 2*v_class_idx) == 0)
            v = self.draw_from_class_velocity(v_class_idx)

    def follow_all_particles(self):
        for i in range(self.n_particles):
            self.follow_one_particle(i)
        print "removing blocked particles: ", len(self.blocked_particles)
        idx_array = np.array(range(self.n_particles))
        blocked = np.array(self.blocked_particles)
        idx_diff = np.setdiff1d(idx_array, blocked)
        self.x_array = self.x_array[idx_diff]
        self.time_array = self.time_array[idx_diff]
        self.n_particles -= len(self.blocked_particles)


class dispModelTime3d(dispersionModelGeneral):
    def __init__(self, n_particles, n_steps, dt, x_max, trans_matrix,
                 mapping, init_class_count, inj_location = "start",
                 verbose = True):
        super(dispModelTime3d,self).__init__(n_particles, n_steps, inj_location, verbose)
        self.trans_matrix = trans_matrix
        self.init_class_count = init_class_count
        self.mapping = mapping
        self.dt = dt
        self.x_max = x_max
        self.init_class_cdf = get_cdf(init_class_count)
        self.n_class = np.sqrt(trans_matrix.shape[0])
        self.blocked_particles = []

    def draw_from_class_velocity(self, idx):
        v_log_edges = self.class_log_edges
        x = random.random()
        log_v = v_log_edges[idx]*x + v_log_edges[idx+1]*(1-x)
        return np.exp(log_v)

    def draw_from_init_calss_idx(self):
        return bs.bisect_right(self.init_class_cdf, random.random())

    def choose_next_class(self, current_class):
        indptr = self.trans_matrix.indptr
        start = indptr[current_class]
        end = indptr[current_class+1]
        rows = self.trans_matrix.indices[start:end]
        values = self.trans_matrix.data[start:end]
        if len(values) == 0:
            return -12
        cdf = get_cdf(values)
        return rows[bs.bisect_left(cdf, random.random())]

    def advance_x_t(self, v, v_sign, freq, x, t):
        dt = self.dt
        dx = v_sign*v*dt
        if freq>1:
            t2 = np.arange(t, t + freq*dt, dt)
            x2 = x + np.arange(1,1+freq)*dx
        else:
            t2 = t + dt
            x2 = x + v_sign*v*dt
        return x2, t2

    def follow_one_particle(self, particle_number):
        x_array = self.x_array
        t_array = self.time_array
        x = 0.0
        t = 0.0
        out_put_idx = 1
        #initialize the particle velocity
        #class_idx is the index of the 2d class
        class_idx = self.draw_from_init_calss_idx()
        idx_max = self.n_steps + 1
        while out_put_idx < idx_max:
            v_1d_class, v_sign, freq = self.mapping.find_absvclass_sgn_freq(class_idx)
            v = self.mapping.draw_from_class_velocity(v_1d_class)
            x_new, t_new = self.advance_x_t(v, v_sign, freq, x, t)
            if freq>1:
                end_idx = min(out_put_idx+freq, idx_max)
                len_idx = end_idx - out_put_idx
                x_array[particle_number, out_put_idx:end_idx] = x_new[:len_idx]
                t_array[particle_number, out_put_idx:end_idx] = t_new[:len_idx]
                out_put_idx += freq
                x = x_new[-1]
                t = t_new[-1]
            else:
                x_array[particle_number, out_put_idx] = x_new
                t_array[particle_number, out_put_idx] = t_new
                out_put_idx += 1
                x = x_new
                t = t_new
            next_idx = self.choose_next_class(class_idx)
            class_idx = next_idx
            if next_idx == -12:
                self.blocked_particles.append(particle_number)
                return

    def follow_all_particles(self):
        for i in range(self.n_particles):
            if not np.mod(i,200):
                print 'particle number: ', i
            self.follow_one_particle(i)
        print "removing blocked particles: ", len(self.blocked_particles)
        idx_array = np.array(range(self.n_particles))
        blocked = np.array(self.blocked_particles)
        idx_diff = np.setdiff1d(idx_array, blocked)
        self.x_array = self.x_array[idx_diff]
        self.time_array = self.time_array[idx_diff]
        self.n_particles -= len(self.blocked_particles)
