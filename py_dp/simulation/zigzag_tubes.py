import numpy as np


class zigzag_tubes(object):
    def __init__(self,nr_t,l,area):
        self.A_tot = area*np.ones(nr_t, dtype=np.float)
        self.l     = l*np.ones(nr_t, dtype=np.float)
