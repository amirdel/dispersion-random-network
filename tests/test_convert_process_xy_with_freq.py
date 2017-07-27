import numpy as np
from py_dp.dispersion.convert_to_time_process_with_freq import get_time_dx_dy_array_with_freq
from py_dp.dispersion.convert_to_time_process_with_freq import remove_duplicate_xy


def test_convert_to_time_process_xyf_1():
    x_start = 0.0
    y_start = 0.0
    dt_array = np.array([10.0, 0.5, 1.6, 0.9], dtype=np.float)
    vx_array = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float)
    vy_array = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float)
    deltaT = 1.0
    dx_array, dy_array, freq_array = get_time_dx_dy_array_with_freq(dt_array, vx_array, vy_array, x_start, y_start, deltaT)
    dx2, dy2, freq2 = remove_duplicate_xy(dx_array, dy_array, freq_array)
    expected_dx = np.array([ 1.0, 2.5, 3.0, 3.9])
    expected_dy = np.array([4.0, 2.5, 2.0, 1.1])
    expected_freq = np.array([ 10., 1., 1., 1.])
    diff_dx_norm = np.linalg.norm(dx2 - expected_dx)
    diff_dy_norm = np.linalg.norm(dy2 - expected_dy)
    print "norm(diff_dx): ", diff_dx_norm
    tol = 1e-12
    assert(diff_dx_norm < tol)
    assert(diff_dy_norm < tol)
    assert(np.all(freq2 == expected_freq))


def test_convert_to_time_process_xyf_2():
    x_start = 0.0
    y_start = 0.0
    dt_array = np.array([1.0, 0.1, 2.0, 0.9], dtype=np.float)
    vx_array = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float)
    vy_array = 3*vx_array
    deltaT = 1.0
    expected_dx   = np.array([ 1.0, 2.9, 3.0, 3.9])
    expected_freq = np.array([ 1., 1., 1., 1.])
    dx_array, dy_array, freq_array = get_time_dx_dy_array_with_freq(dt_array, vx_array, vy_array, x_start, y_start, deltaT)
    diff_dx_norm = np.linalg.norm(dx_array - expected_dx)
    diff_dy_norm = np.linalg.norm(dy_array - 3*expected_dx)
    print "norm(diff_dx): ", diff_dx_norm
    tol = 1e-12
    assert(diff_dx_norm < tol)
    assert(diff_dy_norm < tol)
    assert(np.all(freq_array == expected_freq))


def test_convert_to_time_process_xyf_3():
    x_start = 0.0
    y_start = 0.0
    dt_array = np.array([0.6, 0.6, 4.0], dtype=np.float)
    vx_array = np.array([1.0, 1.0, 1.0], dtype=np.float)
    vy_array = 3*np.array([1.0, 1.0, 1.0], dtype=np.float)
    deltaT = 0.4
    expected_dx   = np.array([ 0.4, 0.4, 0.4, 0.4, 0.4])
    expected_freq = np.array([ 1, 1, 1, 9, 1])
    dx_array, dy_array, freq_array = get_time_dx_dy_array_with_freq(dt_array, vx_array, vy_array, x_start, y_start, deltaT)
    diff_dx_norm = np.linalg.norm(dx_array - expected_dx)
    diff_dy_norm = np.linalg.norm(dy_array - 3*expected_dx)
    print "norm(diff_dx): ", diff_dx_norm
    tol = 1e-12
    assert(diff_dx_norm < tol)
    assert(diff_dy_norm < tol)
    assert(np.all(freq_array == expected_freq))