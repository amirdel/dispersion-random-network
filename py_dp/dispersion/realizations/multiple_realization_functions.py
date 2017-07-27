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

import os
import pickle
from py_dp.simulation.linear_system_solver import LinearSystemStandard, LSGridPeriodicPurturbations
from py_dp.dispersion.dispersion_network import dispersion_system_network
from py_dp.dispersion.dispersion_continua import dispersionSystemContinua
from py_dp.dispersion.dispersion_aux_classes import *

def last_realization_number(save_folder, prefix='real'):
    number_list = []
    for root, dirs, files in os.walk(save_folder):
        for filename in files:
            if filename.startswith(prefix):
                tmp = filename.split('.')[0]
                number_list.append(int(tmp.split('_')[1]))
    return max(number_list)

def save_results_pickle(save_path, x_array, y_array, t_array, last_idx_array=None):
    if last_idx_array == None:
        save_shape = x_array.shape
        last_idx_array = save_shape[1]*np.ones(save_shape[0], dtype=np.int)
    data_holder = dispersionSaver(x_array, t_array, last_idx_array, y_array)
    with open(save_path, 'wb') as output:
        pickle.dump(data_holder, output, pickle.HIGHEST_PROTOCOL)

def save_case_info(save_folder, n_runs, n_particles, n_combine, n_files, n_steps):
    total_n_particles = n_runs*n_particles
    info_file = os.path.join(save_folder, 'case_info.npz')
    np.savez(info_file, total_n_particles=total_n_particles, n_runs=n_runs, n_combine=n_combine, n_files=n_files,
            n_steps = n_steps)

def make_multiple_realizations(network_save_folder, skeleton_name, data_save_folder, inj_type, sigma, p1, p2, n_particles,
                               n_steps, n_runs, bin_width):
    """
    This function would make a number of random network (lognormal) realizations and save it for post processing
    :param network_save_folder: folder of saved network skeleton
    :param skeleton_name: name of the network skeleton
    :param data_save_folder: parent directory to save the resulting realizations e.g. .../zig_zag_500
    :param inj_type: type of injection. can be either 'oneMiddle or 'fluxBased'
    :param sigma: sigma of lognormal distribution
    :param p1: pressure at the left boundary
    :param p2: pressure at the right boundary
    :param n_particles: number of particles in each realization
    :param n_steps: number of jumps saved for each model
    :param n_runs: number of total realizations
    :param bin_width: not used now
    """
    folder_name = inj_type + "_" + str(n_runs)
    sigma_split = str(sigma).split('.')
    if sigma_split[1] == '0':
        sigma_str = sigma_split[0]
    else:
        sigma_str = sigma_split[0] + '_' + sigma_split[1]
    #appending the sigma information to output save path
    save_base = os.path.join(data_save_folder, 'sigma_' + sigma_str)
    if not (os.path.exists(save_base)):
        os.mkdir(save_base)
    #appending the injection type and number of realizations to output save path
    save_base = os.path.join(save_base, folder_name)
    #loading the network skeleton
    network_save_path = os.path.join(network_save_folder, skeleton_name)
    with open(network_save_path, 'r') as input:
        base_network = pickle.load(input)
    # assign random transmissibility to tubes
    base_network.tube_trans_1phase = np.random.lognormal(sigma=sigma,size=base_network.nr_t)
    # apply dirichlet condition at two slabs and solve the pressure system
    pore_x = base_network.pores.x
    base_network_length = np.amax(base_network.pores.x) - np.amin(base_network.pores.x)
    #setting pressure at the two ends of the network
    x1_target = 0.0
    x2_target = base_network_length
    idx1 = np.where(pore_x==x1_target)[0]
    idx2 = np.where(pore_x==x2_target)[0]

    LS = LinearSystemStandard(base_network)
    LS.fill_matrix(base_network.tube_trans_1phase)
    #set the BC
    LS.set_dirichlet_pores(idx1, p1)
    LS.set_dirichlet_pores(idx2, p2)
    #solve for pressure
    LS.solve()
    base_network.pores.p_n[:] = LS.sol
    start_idx = 0
    run_cases = True
    if not (os.path.exists(save_base)):
        os.mkdir(save_base)
    else:
        print "folder already exists. Do you want to continue? [y/n]"
        while True:
            #TODO: changed to work with nohup
            # choice = raw_input()
            choice = 'y'
            if choice == 'y':
                last_idx = last_realization_number(save_base)
                start_idx = last_idx+1    
                print "ok, starting from realization number ", str(start_idx)
                break
            elif choice == 'n':
                print "stopping here!"
                run_cases = False
                break
            else:
                print "invalid choice, try again with 'y' or 'n'"
    if run_cases:
        for i in range(start_idx, n_runs):
            print "--------- realization: ", i, " ----------------"
            base_network.tube_trans_1phase = np.random.lognormal(sigma=sigma, size=base_network.nr_t)
            LS.fill_matrix(base_network.tube_trans_1phase)
            LS.set_dirichlet_pores(idx1,p1)
            LS.set_dirichlet_pores(idx2,p2)
            LS.solve()
            base_network.pores.p_n[:] = LS.sol
            dn1 = dispersion_system_network(base_network, n_particles, n_steps, bin_width, injType=inj_type)
            dn1.follow_all_particles()
            data_holder = dispersionSaver(dn1.x_array, dn1.time_array, dn1.lastIndexArray, y_array = dn1.y_array)
            file_name = "real_" + str(i) + ".pkl"
            save_file = os.path.join(save_base,file_name)
            with open(save_file, 'wb') as output:
                pickle.dump(data_holder, output, pickle.HIGHEST_PROTOCOL)


def multiple_network_relizations(network_path, save_folder, inj_type, sigma, p1, p2, n_particles,
                                 n_steps, n_runs, n_combine):
    """
    This function would make a number of random network (lognormal) realizations and save it for post processing
    :param network_path: full path of saved network skeleton
    :param save_folder: directory to save the resulting realizations
    :param inj_type: type of injection. can be either 'oneMiddle or 'fluxBased'
    :param sigma: sigma of lognormal distribution
    :param p1: pressure at the left boundary
    :param p2: pressure at the right boundary
    :param n_particles: number of particles in each realization
    :param n_steps: number of jumps saved for each model
    :param n_runs: number of total realizations
    :param n_combine: number of realizations to combine for each save file
    """
    with open(network_path, 'r') as input:
        base_network = pickle.load(input)
    # assign random transmissibility to tubes
    base_network.tube_trans_1phase = np.random.lognormal(sigma=sigma,size=base_network.nr_t)
    # apply dirichlet condition at two slabs and solve the pressure system
    pore_x = base_network.pores.x
    base_network_length = np.amax(base_network.pores.x) - np.amin(base_network.pores.x)
    #setting pressure at the two ends of the network
    x1_target = 0.0
    x2_target = base_network_length
    idx1 = np.where(pore_x==x1_target)[0]
    idx2 = np.where(pore_x==x2_target)[0]

    LS = LinearSystemStandard(base_network)
    LS.fill_matrix(base_network.tube_trans_1phase)
    #set the BC
    LS.set_dirichlet_pores(idx1, p1)
    LS.set_dirichlet_pores(idx2, p2)
    #solve for pressure
    LS.solve()
    base_network.pores.p_n[:] = LS.sol
    big_x, big_y, big_t = (np.empty((0, n_steps + 1)) for i in range(3))
    counter = 0
    for i in range(n_runs):
        base_network.tube_trans_1phase = np.random.lognormal(sigma=sigma, size=base_network.nr_t)
        LS.fill_matrix(base_network.tube_trans_1phase)
        LS.set_dirichlet_pores(idx1,p1)
        LS.set_dirichlet_pores(idx2,p2)
        LS.solve()
        base_network.pores.p_n[:] = LS.sol
        dn1 = dispersion_system_network(base_network, n_particles, n_steps, injType=inj_type)
        dn1.follow_all_particles()
        # we want to combine the data for n_combine realizations as save them together
        # for more efficient save and load operations
        # append to what we have sofar
        big_t = np.vstack((big_t, dn1.time_array))
        big_x = np.vstack((big_x, dn1.x_array))
        big_y = np.vstack((big_y, dn1.y_array))
        if n_combine ==1 or (i>0 and (i+1)%n_combine == 0) or i+1 == n_runs:
            #save the arrays
            print 'current realization: ', i, 'saving combined data'
            file_name = 'real_' + str(counter) + '.pkl'
            save_path = os.path.join(save_folder, file_name)
            save_results_pickle(save_path, big_x, big_y, big_t)
            big_x, big_y, big_t = (np.empty((0, n_steps+1)) for i in range(3))
            counter += 1
    # save useful information about the case
    save_case_info(save_folder, n_runs, n_particles, n_combine, counter+1, n_steps)


import pandas as pd
def multiple_continuum_realizations(grid_path, save_folder, perm_path, dp_x, dp_y, n_particles,
                                    n_steps, n_runs, n_combine, n_buffer_ly = 23, ly = 4, std = 1.0):
    # loading the grid
    with open(grid_path, 'r') as input:
        grid = pickle.load(input)
    # load the permeability dataframe, for now reading all of them at the same time
    perm_frame = pd.read_csv(perm_path, usecols=range(n_runs))
    # initialize a linear system for the pressure fluctuations for the grid
    LS = LSGridPeriodicPurturbations(grid)
    # for the number of specified realizations run particle tracking and save the results
    big_x, big_y, big_t = (np.empty((0, n_steps + 1)) for i in range(3))
    counter = 0
    for i in range(n_runs):
        perm = np.exp(std * perm_frame.ix[:, i])
        grid.set_transmissibility(perm)
        # solve for fluctuations around mean pressure gradient
        # setting the left hand side of the equation
        LS.fill_matrix(grid.transmissibility)
        # for each cell add (dp_x/lx)*(T_down - T_up)_x + (dp_y/ly)*(T_down - T_up)_y
        # to the rhs
        rhs_vec = LS.periodic_rhs_vec(dp_x, dp_y)
        LS.rhs.set_neumann_pores_distributed(range(grid.nr_p), rhs_vec)
        # set a dirichlet cell: no fluctuation for cell 0
        LS.set_dirichlet_pores([0], 0.0)
        LS.solve()
        # perform particle tracking
        grid.pressure = LS.sol
        grid.face_velocities = LS.set_face_velocity(dp_x, dp_y)
        u, v = LS.get_cell_velocity()
        dn1 = dispersionSystemContinua(grid, n_particles, n_steps, tracking_type='exit')
        # dn1.init_particles_left_buffered(30)
        # dn1.init_particles_left_boundary()
        dn1.init_particles_ly_distance(n_buffer_ly, ly)
        dn1.follow_all_particles()
        # we want to combine the data for n_combine realizations as save them together
        # for more efficient save and load operations
        if (i > 0 and (i + 1) % n_combine == 0) or i + 1 == n_runs:
            # save the arrays
            print 'current realization: ', i, ', saving combined data'
            file_name = 'real_' + str(counter) + '.pkl'
            save_path = os.path.join(save_folder, file_name)
            # np.savez(save_path, x_array=big_x, y_array=big_y, t_array=big_t)
            save_results_pickle(save_path, big_x, big_y, big_t)
            # initialize the big arrays
            big_x, big_y, big_t = (np.empty((0, n_steps + 1)) for i in range(3))
            counter += 1
            # save useful information about the case
            total_n_particles = n_runs * n_particles
        else:
            # append to what we have sofar
            big_t = np.vstack((big_t, dn1.time_array))
            big_x = np.vstack((big_x, dn1.x_array))
            big_y = np.vstack((big_y, dn1.y_array))
    n_files = counter + 1
    save_case_info(save_folder, n_runs, n_particles, n_combine, n_files, n_steps)