import numpy as np
from py_dp.dispersion.convert_to_time_process_with_freq import get_time_dx_array_with_frequency
import pickle
import os 

def test_convert_to_time_process_with_freq():
   dt_array = np.array([10.0, 0.5, 1.6, 0.9], dtype=np.float)
   v_array = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float)
   deltaT = 1.0
   expected_dx   = np.array([ 1.0, 1.0, 1.0, 2.5, 3.0, 3.9])
   expected_freq = np.array([ 1., 8., 1., 1., 1., 1.])
   dx_array, freq_array = get_time_dx_array_with_frequency(dt_array, v_array, deltaT)
   diff_dx_norm = np.linalg.norm(dx_array - expected_dx)
   print "norm(diff_dx): ", diff_dx_norm
   tol = 1e-12
   assert(diff_dx_norm < tol)
   assert(np.all(freq_array == expected_freq))

def test_convert_to_time_process_with_freq_2():
   dt_array = np.array([1.0, 0.1, 2.0, 0.9], dtype=np.float)
   v_array = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float)
   deltaT = 1.0
   expected_dx   = np.array([ 1.0, 2.9, 3.0, 3.9])
   expected_freq = np.array([ 1., 1., 1., 1.])
   dx_array, freq_array = get_time_dx_array_with_frequency(dt_array, v_array, deltaT)
   diff_dx_norm = np.linalg.norm(dx_array - expected_dx)
   print "norm(diff_dx): ", diff_dx_norm
   tol = 1e-12
   assert(diff_dx_norm < tol)
   assert(np.all(freq_array == expected_freq))

def test_convert_to_time_process_with_freq_3():
   dt_array = np.array([0.6, 0.6, 4.0], dtype=np.float)
   v_array = np.array([1.0, 1.0, 1.0], dtype=np.float)
   deltaT = 0.4
   expected_dx   = np.array([ 0.4, 0.4, 0.4, 0.4, 0.4])
   expected_freq = np.array([ 1, 1, 1, 9, 1])
   dx_array, freq_array = get_time_dx_array_with_frequency(dt_array, v_array, deltaT)
   print dx_array
   print freq_array
   diff_dx_norm = np.linalg.norm(dx_array - expected_dx)
   print "norm(diff_dx): ", diff_dx_norm
   tol = 1e-12
   assert(diff_dx_norm < tol)
   assert(np.all(freq_array == expected_freq))

def test_particle_tracking_results_conversion():
    import bisect as bs
    current_dir = os.getcwd()
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    p_track_file = 'test_related_files/particle_tracking_results/real_0.pkl'
    input_address = os.path.join(parent_dir, p_track_file)
    with open(input_address,'rw') as input:
        dataHolder = pickle.load(input)
    x_data = dataHolder.x_array
    t_data = dataHolder.t_array
    #check the length of the path for all particles
    for deltaT in [1.0, 20.0, 50.0]:
        for i in range(x_data.shape[0]):
            x_array = x_data[i,:]
            dt_array = np.diff(t_data[i,:])
            dx_array = np.diff(x_array)
            dt_filter = dt_array != 0.0
            dt_input = dt_array[dt_filter]
            dx_input = dx_array[dt_filter]
            v_input = np.divide(dx_input, dt_input)
            length_from_file = np.sum(dx_input)
            time_from_file = np.sum(dt_input)
            dx_output, freq_output = get_time_dx_array_with_frequency(dt_input, v_input,deltaT)
            if len(dx_output) >0:
            #in this test all particles are released at zero and the domain in not periodic in x,
            #and is periodic in y
                assert(dx_output[0]>0)
                t_end_model = len(dx_output)*deltaT
                idx_end = bs.bisect_left(np.cumsum(dt_input), t_end_model)
                length_from_function = np.sum(np.multiply(dx_output, freq_output))
                diff_dx = length_from_file - length_from_function
                max_expected_diff = np.amax(v_input[idx_end-1:])*deltaT
                assert(abs(diff_dx) <= abs(max_expected_diff))
