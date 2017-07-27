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
from py_dp.simulation.zigzag_pores import zigzag_pores
from py_dp.simulation.zigzag_tubes import zigzag_tubes

class zigzag(object):
    def __init__(self,m,n,l,theta,boundaryType="periodic"):
        """
        Zigzag networks that have all the useful fields of a pore network. 
        If periodic, the network is only periodic in the y direction.
        """
        self.m = m
        self.n = n
        self.nr_p = m*n
        self.l = l
        self.theta = theta
        self.boundaryType = boundaryType
        self.ngh_pores = -1*np.ones(self.nr_p, dtype=np.object)
        self.ngh_tubes = -1*np.ones(self.nr_p, dtype=np.object)
        self.make_pore_connectivity_list()
        self.make_pore_tube_connectivity()
        dx = l*np.cos(theta)
        dy = l*np.sin(theta)
        dz = 0.0
        self.pores = zigzag_pores(m,n,dx,dy,dz)
        self.tube_trans_1phase = np.ones(self.nr_t, dtype=float)
        self.nr_nghs = self.create_nr_nghs() 
        area = 1.0
        self.tubes = zigzag_tubes(self.nr_t,l,area)
        self.network_length, self.network_width = self.network_dim()

    def network_dim(self):
        xarray = self.pores.x
        yarray = self.pores.y
        dx = np.amax(xarray) - np.amin(xarray)
        dy = np.amax(yarray) - np.amin(yarray)
        return dx,dy

    def create_nr_nghs(self):
        nr_ngh = 4*np.ones(self.nr_p, dtype=int)
        n = self.n
        m = self.m
        for i in (np.array(range(m))+1):
            #first column
            pn = i -1
            nr_ngh[pn] -= 2
            #last column
            pn = (n-1)*m + i -1
            nr_ngh[pn] -= 2
        if (self.boundaryType != "periodic"):
            for j in (np.array(range(n))+1):
                if (j%2 == 1):
                #first row
                    pn = (j-1)*m
                    nr_ngh[pn] -= 2 
                    nr_ngh[pn] += (pn == 0) + (pn == (n-1)*m)
                else:
                #last row
                    pn = (j-1)*m + m-1
                    nr_ngh[pn] -= 2 
                    nr_ngh[pn] += 2*(pn == (m-1)) + (pn == (n*m-1))
        return nr_ngh

    def make_pore_connectivity_list(self):
        m = self.m 
        n = self.n
        nr_p = self.nr_p
        p_adj = -1*np.ones([nr_p,4],dtype=int)
        if (self.boundaryType != "periodic"):
            for j in (np.array(range(n))+1):
                for i in (np.array(range(m))+1):
                    pn = (j-1)*m + i -1
                    if np.mod(j,2) == 1:
                        p_adj[pn,:] = [(pn-m-1)*(j>1)*(i>1) + (-1)*(j==1) + (-1)*(i==1),
                                       (pn-m)*(j>1) + (-1)*(j==1),
                                       (pn+m-1)*(i>1)*(j<n) + (-1)*(j==n) + (-1)*(i==1),
                                       (pn+m)*(j<n) + (-1)*(j==n)]
                    else:
                        p_adj[pn,:] = [(pn-m)*(j>1) + (-1)*(j==1),
                                       (pn-m+1)*(j>1)*(i<m) + (-1)*(j==1) + (-1)*(i==m),
                                       (pn+m)*(j<n) + (-1)*(j==n),
                                       (pn+m+1)*(j<n)*(i<m) + (-1)*(j==n) + (-1)*(i==m)]
        else:
            for j in (np.array(range(n))+1):
                for i in (np.array(range(m))+1):
                    pn = (j-1)*m + i -1
                    if np.mod(j,2) == 1:
                        p_adj[pn,:] = [(pn-m-1)*(j>1)*(i>1) + (-1)*(j==1) + (j!=1)*(i==1)*(pn-1),
                                       (pn-m)*(j>1) + (-1)*(j==1),
                                       (pn+m-1)*(i>1)*(j<n) + (-1)*(j==n) + (j!=n)*(i==1)*(pn+2*m-1),
                                       (pn+m)*(j<n) + (-1)*(j==n)]
                    else:
                        p_adj[pn,:] = [(pn-m)*(j>1) + (-1)*(j==1),
                                       (pn-m+1)*(j>1)*(i<m) + (-1)*(j==1) + (j!=1)*(i==m)*(pn-2*m+1),
                                       (pn+m)*(j<n) + (-1)*(j==n),
                                       (pn+m+1)*(j<n)*(i<m) + (-1)*(j==n) + (j!=n)*(i==m)*(pn+1)]
        for i in range(self.nr_p):
            self.ngh_pores[i] = p_adj[i][p_adj[i] > -1]
        del p_adj

    def make_pore_tube_connectivity(self):
        nr_p = self.nr_p
        ngh_pores = self.ngh_pores
        tp_adj = np.array([-1,-1])
        tube_dict = {}
        tubeNumber = 0
        for i in range(nr_p):
            nghArray = ngh_pores[i]
            nghBigger = nghArray[nghArray>i]
            for ngh in nghBigger:
                temp = [i,ngh]
                key = (i,ngh)
                tp_adj = np.vstack((tp_adj,temp))
                tube_dict[key] = tubeNumber
                tubeNumber += 1
        self.tp_adj = tp_adj[1:,:]
        t_adj = -1*np.ones(nr_p,dtype=np.object)
        for i in range(nr_p):
            nghArray = ngh_pores[i]
            for j in nghArray:
                t_adj[i] = np.append(t_adj[i],tube_dict[tuple(sorted((i,j)))])
        for i in range(nr_p):
            self.ngh_tubes[i] = t_adj[i][t_adj[i] > -1]
        self.nr_t = tubeNumber
        del tube_dict
        del t_adj

    def get_pore_mask_from_pore_indices(self, pi_list):
        pore_mask = np.zeros(self.nr_p, dtype=np.bool)
        pore_mask[pi_list] = True
        return pore_mask
