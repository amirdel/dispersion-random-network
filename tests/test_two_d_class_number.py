from py_dp.dispersion.second_order_markov import find_1d_bins, find_2d_bin

def test_two_d_class_number():
    n_class = 4.0
    c1 = 0.0
    c2 = 0.0
    c_2d = find_2d_bin(c1, c2, n_class)
    assert(c_2d == 0)
    c1b, c2b = find_1d_bins(c_2d, n_class)
    assert(c1b==c1 and c2b==c2)
    c1 = 1.0
    c2 = 2.0
    c_2d = find_2d_bin(c1, c2, n_class)
    assert(c_2d == 6)
    c1b, c2b = find_1d_bins(c_2d, n_class)
    assert(c1b==c1 and c2b==c2)
    c1 = 1.0
    c2 = 2.0
    c_2d = find_2d_bin(c1, c2, n_class)
    assert(c_2d == 6)
    c1b, c2b = find_1d_bins(c_2d, n_class)
    assert(c1b==c1 and c2b==c2)
    c1 = 3.0
    c2 = 3.0
    c_2d = find_2d_bin(c1, c2, n_class)
    assert(c_2d == 15)
    c1b, c2b = find_1d_bins(c_2d, n_class)
    assert(c1b==c1 and c2b==c2)
