import numpy as np
cimport numpy as np
DTYPE = np.int
ctypedef np.int_t DTYPE_t
DTYPE2 = np.float
ctypedef np.float_t DTYPE2_t
import bisect as bs

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function

def time_avg_dx_dy_with_freq_cython(np.ndarray[DTYPE2_t] dt_input, np.ndarray[DTYPE2_t] vx_input,
                                    np.ndarray[DTYPE2_t] vy_input, DTYPE2_t x_start, DTYPE2_t y_start,
                                    DTYPE2_t delta_t):
    """
    function to find the dx, dy, number of repeats array for the in time with dt increments of deltaT
    output theta will be in radians form -pi to pi
    :param dt_input:
    :param vx_input:
    :param vy_input:
    :param x_start:
    :param y_start:
    :param delta_t:
    :return:
    """
    freq_list = []
    dx_list = []
    dy_list = []
    cdef np.ndarray[DTYPE2_t] t_array, x_array, y_array
    t_array = np.hstack(([0.0], np.cumsum(dt_input)))
    cdef DTYPE2_t end_time = t_array[len(t_array)-1]
    ## x_array , y_array include the x,y location of the particle
    x_array = x_start + np.hstack((0.0, np.cumsum(np.multiply(dt_input, vx_input))))
    y_array = y_start + np.hstack((0.0, np.cumsum(np.multiply(dt_input, vy_input))))
    cdef DTYPE2_t t_target, closest_smaller_time, dx, x_correction, dy, y_correction
    cdef DTYPE2_t closest_larger_time, repeating_dx, repeating_dy
    cdef int n_repeat
    t_target = delta_t
    while t_target <= end_time:
        idx_t = bs.bisect_left(t_array, t_target)
        closest_smaller_time = t_array[idx_t -1]
        #advance x
        dx = x_array[idx_t-1] - x_start
        x_correction = (t_target - closest_smaller_time)*vx_input[idx_t - 1]
        dx += x_correction
        dx_list.append(dx)
        x_start += dx
        #advance y
        dy = y_array[idx_t - 1] - y_start
        y_correction = (t_target - closest_smaller_time) * vy_input[idx_t - 1]
        dy += y_correction
        dy_list.append(dy)
        y_start += dy
        #add one to frequency
        freq_list.append(1)
        t_target += delta_t
        #take care of repetition
        closest_larger_time = t_array[idx_t]
        n_repeat = np.floor((closest_larger_time - t_target)/delta_t)
        if n_repeat > 0:
            repeating_dx = vx_input[idx_t - 1]*delta_t
            repeating_dy = vy_input[idx_t - 1]*delta_t
            dx_list.append(repeating_dx)
            dy_list.append(repeating_dy)
            freq_list.append(n_repeat)
            x_start += n_repeat*repeating_dx
            y_start += n_repeat*repeating_dy
            t_target += n_repeat*delta_t
    return dx_list, dy_list, freq_list


def time_avg_dx_dy_with_freq_cython2(np.ndarray[DTYPE2_t] dt_input, np.ndarray[DTYPE2_t] vx_input,
                                    np.ndarray[DTYPE2_t] vy_input, DTYPE2_t x_start, DTYPE2_t y_start,
                                    DTYPE2_t delta_t):
    """
    function to find the dx, dy, number of repeats array for the in time with dt increments of deltaT
    output theta will be in radians form -pi to pi
    :param dt_input:
    :param vx_input:
    :param vy_input:
    :param x_start:
    :param y_start:
    :param delta_t:
    :return:
    """
    freq_list = []
    dx_list = []
    dy_list = []
    cdef np.ndarray[DTYPE2_t] t_array, x_array, y_array
    t_array = np.hstack(([0.0], np.cumsum(dt_input)))
    cdef DTYPE2_t end_time = t_array[len(t_array)-1]
    cdef int arr_size
    arr_size = int(2*end_time/delta_t) + 1
    cdef np.ndarray[DTYPE2_t] dx_array, dy_array
    cdef np.ndarray[DTYPE_t] freq_array
    dx_array = np.zeros(arr_size, dtype = DTYPE2)
    dy_array = np.zeros(arr_size, dtype = DTYPE2)
    freq_array = np.zeros(arr_size, dtype = DTYPE)
    ## x_array , y_array include the x,y location of the particle
    x_array = x_start + np.hstack((0.0, np.cumsum(np.multiply(dt_input, vx_input))))
    y_array = y_start + np.hstack((0.0, np.cumsum(np.multiply(dt_input, vy_input))))
    cdef DTYPE2_t t_target, closest_smaller_time, dx, x_correction, dy, y_correction
    cdef DTYPE2_t closest_larger_time, repeating_dx, repeating_dy
    cdef int n_repeat, front
    front = 0
    t_target = delta_t
    while t_target <= end_time:
        idx_t = bs.bisect_left(t_array, t_target)
        closest_smaller_time = t_array[idx_t -1]
        #advance x
        dx = x_array[idx_t-1] - x_start
        x_correction = (t_target - closest_smaller_time)*vx_input[idx_t - 1]
        dx += x_correction
        dx_array[front] = dx
        x_start += dx
        #advance y
        dy = y_array[idx_t - 1] - y_start
        y_correction = (t_target - closest_smaller_time) * vy_input[idx_t - 1]
        dy += y_correction
        dy_array[front] = dy
        y_start += dy
        #add one to frequency
        freq_array[front] = 1
        front += 1
        t_target += delta_t
        #take care of repetition
        closest_larger_time = t_array[idx_t]
        n_repeat = int((closest_larger_time - t_target)/delta_t)
        if n_repeat > 0:
            repeating_dx = vx_input[idx_t - 1]*delta_t
            repeating_dy = vy_input[idx_t - 1]*delta_t
            dx_array[front] = repeating_dx
            dy_array[front] = repeating_dy
            freq_array[front] = n_repeat
            front += 1
            x_start += n_repeat*repeating_dx
            y_start += n_repeat*repeating_dy
            t_target += n_repeat*delta_t
    return dx_array[:front], dy_array[:front], freq_array[:front]