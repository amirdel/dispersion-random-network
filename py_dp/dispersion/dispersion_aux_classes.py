import numpy as np
from copy import deepcopy
import pickle as pickle
import os


class dispersionSaver:
    def __init__(self, x_array, t_array, last_idx_array, y_array = None):
        self.x_array = x_array
        self.t_array = t_array
        self.last_idx_array = last_idx_array
        if y_array is not None:
            self.y_array = y_array

class caseInfo(object):
    def __init__(self, inj_type, n_realizations, network_name, max_dx, min_dx, max_v, min_v, max_dt, min_dt):
        self.inj_type = inj_type
        self.n_realizations = n_realizations
        self.network_name = network_name
        self.max_velocity = max_v
        self.min_velocity = min_v
        self.max_dx = max_dx
        self.min_dx = min_dx
        self.max_dt = max_dt
        self.min_dt = min_dt

def find_dispersion_object_min_max(dn1):
    dx = np.diff(dn1.x_array)
    dt = np.diff(dn1.time_array)
    max_dt = np.amax(dt)
    dt[dt == 0.0] = max_dt
    v_array = np.divide(dx, dt)
    v_array[np.isnan(v_array)] = 0.0
    max_dx = np.amax(dx)
    min_dx = np.amin(dx)
    min_dt = np.amin(dt)
    max_v = np.amax(v_array)
    min_v = np.amin(v_array)
    return max_v, min_v, max_dx, min_dx, max_dt, min_dt

def choose_extremes(max_v, max_v1, min_v, min_v1, max_dx, max_dx1, min_dx, min_dx1,
                    max_dt, max_dt1, min_dt, min_dt1):
    max_v = max(max_v, max_v1)
    min_v = min(min_v, min_v1)
    max_dx = max(max_dx, max_dx1)
    min_dx = min(min_dx, min_dx1)
    max_dt = max(max_dt, max_dt1)
    min_dt = min(min_dt, min_dt1)
    return max_v, min_v, max_dx, min_dx, max_dt, min_dt

class correlatedSaver(object):
    def __init__(self, trans_matrix, class_velocity, init_class_count,
                 dt, n_neg_class, n_realizations):
        self.trans_matrix = trans_matrix
        self.class_velocity = class_velocity
        self.init_class_count = init_class_count
        self.dt = dt
        self.n_neg_class = n_neg_class
        self.n_realizations = n_realizations

class correlatedSaver3d(object):
     def __init__(self, trans_matrix, init_class_count, dt, mapping, n_realizations,
                  n_binning_realz, n_slow_classes):
        self.trans_matrix = trans_matrix
        self.mapping = mapping
        self.init_class_count = init_class_count
        self.dt = dt
        self.n_binning_realz = n_binning_realz
        self.n_realizations = n_realizations
        self.n_slow_classes = n_slow_classes

def purturb_network(network, multip):
    network2 = deepcopy(network)
    l = network2.l
    theta = network2.theta
    dx = l*np.cos(theta)
    dy = l*np.sin(theta)
    x_array = network2.pores.x
    y_array = network2.pores.y
    left_boundary = (x_array == 0)
    right_boundary = (x_array == np.amax(x_array))
    inner_cells = ~np.logical_or(left_boundary, right_boundary)
    n_inner_cells = len(np.where(inner_cells)[0])
    rand_array = np.random.uniform(-1,1,n_inner_cells)
    x_array[inner_cells] += multip*dx*rand_array
    rand_array = np.random.uniform(-1,1,n_inner_cells)
    y_array[inner_cells] += multip*dy*rand_array
    return network2

def tube_length(network):
    tp_adj = network.tp_adj
    x_array = network.pores.x
    y_array = network.pores.y
    x1 = x_array[tp_adj[:,0]]
    y1 = y_array[tp_adj[:,0]]
    x2 = x_array[tp_adj[:,1]]
    y2 = y_array[tp_adj[:,1]]
    tube_l_array = np.sqrt(np.power(x2-x1,2) + np.power(y2-y1,2))
    return tube_l_array

def save_results_pickle(save_path, x_array, y_array, t_array, last_idx_array=None):
    if last_idx_array == None:
        save_shape = x_array.shape
        last_idx_array = save_shape[1]*np.ones(save_shape[0], dtype=np.int)
    data_holder = dispersionSaver(x_array, t_array, last_idx_array, y_array)
    with open(save_path, 'wb') as output:
        pickle.dump(data_holder, output, pickle.HIGHEST_PROTOCOL)

def save_case_info(save_folder, n_runs, n_particles, n_combine, n_files):
    total_n_particles = n_runs*n_particles
    info_file = os.path.join(save_folder, 'case_info.npz')
    np.savez(info_file, total_n_particles=total_n_particles, n_runs=n_runs, n_combine=n_combine, n_files=n_files)