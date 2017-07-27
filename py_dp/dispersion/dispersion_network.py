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

import numpy as np
import bisect as bs
import random as random
from py_dp.simulation.network_aux_functions import calc_flux
from py_dp.dispersion.trajectory_count_cython import get_cdf_cython

class dispersion_system_network:
    """
    class to define a dispersion problem on a pore_network
    """
    def __init__(self, network, n_particles, n_steps, binSize=None, injType="fluxBased", verbose=True,
                 remove_all_blocked = False):
        """
        :param network: an instance of a pore network or zigzag
        :param n_particles: number of particles to inject in the network
        :param n_steps: number of steps to follow the particles
        :param binSize: bin size that can be used for specifying an injection area
        :param injType: fluxBased injection into the pores vs. injecting all particles in the middle
        :param verbose: flag to write more output messages
        :param remove_all_blocked: flag to remove all the particles that were blocked (stuck in min pressure pore)
        """
        self.verbose = verbose
        self.network = network
        self.p = network.pores.p_n
        self.pmin = np.amin(self.p)
        self.n_particles = n_particles
        self.n_steps = n_steps
        self.init_pores = np.zeros(n_particles)
        self.time_array    = np.zeros((n_particles,n_steps+1))
        self.pore_nr_array = np.zeros((n_particles,n_steps+1), dtype=np.int)
        self.x_array       = np.zeros((n_particles,n_steps+1))
        self.y_array       = np.zeros((n_particles,n_steps+1))
        self.tetha_array   = np.zeros((n_particles,n_steps+1))
        #self.init_type = init_type
        p_array = network.pores.p_n
        trans_array = network.tube_trans_1phase
        tp_adj = network.tp_adj
        nr_t = network.nr_t
        self.flux_array = calc_flux(p_array,trans_array,tp_adj,nr_t)
        self.init_arrays(binSize, injType)
        self.going_back = 0
        idx_pmax = np.argmax(p_array)
        self.xpmax = self.network.pores.x[idx_pmax]
        #frozen particles are the ones that are inside the pore with a local pressure minimum.
        self.freeze_array = np.zeros(n_particles, dtype=bool)
        self.freeze_time = np.zeros(n_particles, dtype=float)
        self.n_jumps = float(n_particles*n_steps)
        self.avgBackTime = 0.0
        self.dxArray = np.array([])
        self.tArray  = np.array([])
        self.lastIndexArray = n_steps*np.ones(n_particles, dtype=np.int8)
        self.remove_all_blocked = remove_all_blocked
        
    def init_arrays(self, binSize, injType , injLocation="start"):
        if (injLocation not in ["start","shifted"]):
            raise "injLocation can only be start or shifted"
        if (injType not in ["fluxBased","random","oneMiddle"]):
            raise "injType can only be fluxBased or random or oneMiddle"
        network = self.network
        x_array = network.pores.x
        y_array = network.pores.y
        #TODO: add more option for injection location
        slab_pores = np.where(x_array == 0.0)[0]
        if self.verbose:
            print "number of candidates are: ", len(slab_pores)
        if injType == "random":
            init_pores = np.random.choice(slab_pores, self.n_particles, replace=True)
        elif injType == "oneMiddle":
            yArray = network.pores.y[slab_pores]
            ymid = np.amax(yArray)/2
            dist = abs(yArray-ymid)
            idxChosen = np.argmin(dist)
            init_pores = np.array(idxChosen*np.ones(self.n_particles,dtype=int))
        elif injType == "fluxBased":
            pore_list = slab_pores
            p2_list = np.array([],dtype=int)
            for i in pore_list:
                p2_list = np.append(p2_list,i*np.ones_like(network.ngh_tubes[i]))
            tube_list = np.hstack(network.ngh_tubes[pore_list])
            flux_array = self.flux_array[tube_list]
            flux_array /= np.sum(flux_array)
            cdf = [flux_array[0]]
            for i in xrange(1, len(flux_array)):
                cdf.append(cdf[-1]+flux_array[i])
            rnd = np.random.random_sample(self.n_particles)
            bb = np.searchsorted(cdf,rnd)
            init_pores = p2_list[bb]
            #print init_pores
        self.x_array[:,0] = x_array[init_pores]
        self.y_array[:,0] = y_array[init_pores]
        self.pore_nr_array[:,0] = init_pores
        self.init_pores = init_pores
        
            
    @staticmethod
    def draw_based_on_prob(input1):
        input_array = input1[:]
        cdf = get_cdf_cython(input_array)
        random_ind = bs.bisect(cdf,random.random())
        return random_ind

    def advance_to_next_pore(self,current_pore,current_t,particle_number,current_idx):
        flux_array = self.flux_array
        pmin_flag = False
        ngh_pores_array = self.network.ngh_pores
        ngh_tubes_array = self.network.ngh_tubes
        x_array = self.network.pores.x
        area_array = self.network.tubes.A_tot
        length_array = self.network.tubes.l
        ngh_pores = ngh_pores_array[current_pore]
        ngh_tubes = ngh_tubes_array[current_pore]
        current_p = self.p[current_pore]
        ngh_p = self.p[ngh_pores]
        lower_p_mask = (ngh_p<current_p)
        lower_p_pores = ngh_pores[lower_p_mask]
        lower_p_tubes = ngh_tubes[lower_p_mask]
        lower_p_vals = ngh_p[lower_p_mask]
        if (len(lower_p_vals)==0):
            pmin_flag = True
            self.freeze_array[particle_number] = True
            return pmin_flag
        lower_flux = flux_array[lower_p_tubes]
        index = self.draw_based_on_prob(lower_flux)
        next_pore = lower_p_pores[index]
        next_tube = lower_p_tubes[index]
        next_idx = current_idx + 1
        next_x = x_array[next_pore]
        current_x = x_array[current_pore]
        dx = current_x - next_x
        area = area_array[next_tube]
        length = length_array[next_tube]
        velocity = lower_flux[index]/area
        dt = length/velocity
        self.save_jump_info(current_t, dt, particle_number, next_idx, next_pore, next_tube)
        if (next_x<=current_x):
            self.going_back += 1
            self.avgBackTime += dt
        #self.time_array[particle_number,next_idx] = current_t+dt
        #self.x_array[particle_number,next_idx] = next_x
        #self.pore_nr_array[particle_number,next_idx] = next_pore
        return pmin_flag

    def save_jump_info(self, current_t, dt, particle_number, next_idx, next_pore, next_tube):
        self.time_array[particle_number,next_idx] = current_t + dt
        self.x_array[particle_number,next_idx] = self.network.pores.x[next_pore]
        self.y_array[particle_number,next_idx] = self.network.pores.y[next_pore]
        self.pore_nr_array[particle_number,next_idx] = next_pore
        
    def calc_jump_angles(self):
        theta = self.network.theta
        x_array = self.x_array
        y_array = self.y_array
        last_idx_array = self.lastIndexArray
        tetha_array = self.tetha_array
        for particle in range(self.n_particles):
            for i in range(last_idx_array[particle]-1):
                dx = x_array[particle, i+1] - x_array[particle, i]
                dy = y_array[particle, i+1] - y_array[particle, i]
                tetha_array[particle, i] = (dx > 0)*(dy > 0)*theta + (dx > 0)*(dy < 0)*(-theta) +\
                                           (dx < 0)*(dy > 0)*(np.pi - theta) + (dx < 0)*(dy < 0)*(np.pi + theta)

    def follow_specific_paticle(self,particle_number):
        n_steps = self.n_steps
        for step in range(n_steps):
            #x_array, pore_nr_array, time_array entries are changed inside
            #advance_to_next_pore
            current_pore = self.pore_nr_array[particle_number,step]
            current_t = self.time_array[particle_number,step]
            current_index = step
            pmin_flag = self.advance_to_next_pore(current_pore,current_t,particle_number,current_index)
            if pmin_flag:
                self.freeze_particle_in_pore(particle_number,current_index)
                break
    
    def follow_all_particles(self):
        n_particles = self.n_particles
        for particle in range(n_particles):
            self.follow_specific_paticle(particle)
        #now check what is the maximum time --> use that time for 
        if self.going_back > 0:
            if self.verbose:
                print self.going_back, " jump are going backwards"
                print 100*self.going_back/float(self.n_jumps), "% of total jumps"
        freeze = self.freeze_array
        if self.verbose:
            print "number of frozen particles: ", len(np.where(freeze)[0])
        #find if time array is all zeros for any particle
        last_time = self.time_array[:,-1]
        idx_zero = np.where(last_time==0)[0]
        self.x_array = np.delete(self.x_array, idx_zero, 0)
        self.time_array = np.delete(self.time_array, idx_zero, 0)
        self.n_particles -= len(idx_zero)
        self.freeze_array = np.delete(self.freeze_array, idx_zero)
        self.lastIndexArray = np.delete(self.lastIndexArray, idx_zero)
        if self.verbose:
            print len(idx_zero), " particles were removed because they were blocked from initialization"
        if self.verbose:
            if self.going_back > 0:
                self.avgBackTime /= self.going_back
                print "average going back time is: ", '{:e}'.format(self.avgBackTime)
        if self.remove_all_blocked:
            frozen_idx = np.where(self.freeze_array)[0]
            frozen_x = self.x_array[frozen_idx, self.lastIndexArray[frozen_idx]]
            xmax = np.amax(self.network.pores.x) - 1
            target_set = frozen_idx[frozen_x < xmax]
            self.remove_blocked(target_set)

    def remove_blocked(self, target_set):
        n_remove = len(target_set)
        self.x_array = np.delete(self.x_array, target_set, 0)
        self.y_array = np.delete(self.y_array, target_set, 0)
        self.time_array = np.delete(self.time_array, target_set, 0)
        self.n_particles -= n_remove
        self.freeze_array = np.delete(self.freeze_array, target_set)
        self.lastIndexArray = np.delete(self.lastIndexArray, target_set)
        if n_remove:
            print n_remove, " particles were removed because they were blocked in simulation"
        

    def freeze_particle_in_pore(self,particle_number,current_index):
        """
        after a particle gets to a pore with minimum pressure value it would stay there
        """
        self.x_array[particle_number,current_index:] = self.x_array[particle_number,current_index]
        self.pore_nr_array[particle_number,current_index:] = self.pore_nr_array[particle_number,current_index]
        self.time_array[particle_number,current_index:] = self.time_array[particle_number,current_index]
        self.freeze_time[particle_number] = self.time_array[particle_number,current_index]
        self.lastIndexArray[particle_number] = current_index
        tmp = self.n_steps - current_index
        self.n_jumps -= tmp
    
    def plumeLocationAtGivenTime(self,t):
        """
        returns the interpolated location of all particles at time t in a vector 
        """
        time_array = self.time_array
        t_max = np.amin(time_array[:,-1])
        if t>t_max:#not in limits:
            if (self.n_particles == len(np.where(self.freeze_array)[0])):
                print "ok since all particles are in the sink..."
            else:
                raise Exception("time is larger than simulation time!")
        x_array = self.x_array
        n_particles = self.n_particles
        return_x = np.zeros(n_particles)
        n_steps = self.n_steps
        assert(n_particles==len(return_x))
        for p in range(n_particles):
            t_temp = time_array[p,:]
            x_temp = x_array[p,:]
            #index of array member to the left
            idx = bs.bisect_left(t_temp,t)
            if idx >= n_steps:
                return_x[p] = x_temp[-1]
            elif idx == 0:
                return_x[p] = x_temp[0]
            else:
                temp1 = t_temp[idx-1:idx+1]
                temp2 = x_temp[idx-1:idx+1]
                return_x[p] = np.interp(t, temp1, temp2)
        return return_x
    
    def COM_MSD_atGivenTime(self,t):
        """
        return the COM and Mean Square Difference with respect 
        to the center of mass.
        """
        time_array = self.time_array
        t_max = np.amax(time_array[:,-1])
        if t>t_max:#not in limits:
            raise Exception("time is larger than simulation time!")
        plumeLocation = self.plumeLocationAtGivenTime(t)
        com = np.mean(plumeLocation)
        msd = np.mean(np.power(plumeLocation-com,2))
        return com, msd
    
    def timeToGetToLocation(self,x):
        time_array = self.time_array
        x_array = self.x_array 
        x_max = np.amax(x_array[:,-1])
        n_particles = self.n_particles
        return_t = np.zeros(n_particles)
        if x > x_max:
            raise Exception("x bigger than the allowed value!")
        n_steps = self.n_steps
        for p in range(n_particles):
            t_temp = time_array[p,:]
            x_temp = x_array[p,:]
            idx = bs.bisect_left(x_temp, x)
            if idx > n_steps:
                retrun_t[p] = t_temp[-1]
            else:
                return_t[p] = t_temp[idx]
        return return_t
