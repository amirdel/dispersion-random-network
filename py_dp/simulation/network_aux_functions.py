import numpy as np

def calc_flux(p_array,trans_array,tp_adj,nr_t):
    idx1_array = tp_adj[:,0]
    idx2_array = tp_adj[:,1]
    dp = abs(p_array[idx1_array]-p_array[idx2_array])
    flux = np.multiply(trans_array,dp)
    return flux