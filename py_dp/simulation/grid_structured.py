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
from py_dp.simulation.structured_cells import structured_cells
from py_dp.simulation.zigzag_tubes import zigzag_tubes

class structuredGrid(object):
    def __init__(self, m, n, dx, dy, boundaryType='non-periodic'):
        """
        Zigzag networks that have all the useful fields of a pore network.
        If periodic, the network is only periodic in the y direction.
        :param m: number of cells in the y direction
        :param n: number of cells in the x direction
        :param dx: delta x
        :param dy: delta y
        :param boundaryType: can be 'non-periodic' or 'full-periodic'
        """
        #only supporting equal dx and dy for now
        assert(boundaryType in ['full-periodic', 'non-periodic'])
        assert(dx == dy)
        self.m = m
        self.n = n
        self.dx = dx
        self.dy = dy
        #set dz to be equal to dy
        self.dz = dy
        self.lx = n*dx
        self.ly = m*dy
        #number of pores/cells
        self.nr_p = m*n
        self.boundaryType = boundaryType
        #neighbor initialization
        self.ngh_pores = -1*np.ones(self.nr_p, dtype=np.object)
        self.ngh_tubes = -1*np.ones(self.nr_p, dtype=np.object)
        self.make_pore_connectivity_list()
        self.make_pore_tube_connectivity()
        self.y_faces, self.d_cell_numbers = self.make_y_faces_array()
        self.updwn_cells = self.make_updwn_cells()
        self.facelist_array, self.nghlist_array = self.make_face_list_for_cells()
        dz = 0.0
        self.pores = structured_cells(m,n,dx,dy,dz)
        self.transmissibility = np.ones(self.nr_t, dtype=float)
        self.nr_nghs = self.create_nr_nghs() 
        area = 1.0
        self.tubes = zigzag_tubes(self.nr_t, dx, dx*dy)
        self.network_length, self.network_width = self.network_dim()
        self.pressure = None
        self.face_velocities = None
        self.periodic_faces = set()
        if boundaryType == 'full-periodic':
            self.periodic_faces = self.set_periodic_faces()


    def network_dim(self):
        xarray = self.pores.x
        yarray = self.pores.y
        dx = np.amax(xarray) - np.amin(xarray)
        dy = np.amax(yarray) - np.amin(yarray)
        return dx,dy

    def create_nr_nghs(self):
        """
        finding the number of neighbors for each cell
        :return: nr_ngh array (size number of cells nr_p)
        """
        nr_p = self.nr_p
        ngh_cells = self.ngh_pores
        nr_ngh = np.zeros(nr_p, dtype=np.int)
        for i in range(nr_p):
            nr_ngh[i] = len(ngh_cells[i])
        return nr_ngh

    def make_pore_connectivity_list(self):
        """
        make the connectivity list for the cells, different for the periodic and non-periodic
        cases
        :return:
        """
        m = self.m 
        n = self.n
        nCells = self.nr_p
        #order of neighbors: left, down, right, top
        p_adj = -1 * np.ones([nCells, 4], dtype=int)
        self.ngh_pores = -1 * np.ones(nCells, dtype=np.object)
        if self.boundaryType == 'non-periodic':
            for j in (np.array(range(n)) + 1): #from 1 to n
                for i in (np.array(range(m)) + 1): #from 1 to m
                    pn = (j - 1) * m + i - 1
                    #                      inner cells     boundary cell
                    p_adj[pn, :] = [(pn - m) * (j > 1) + (-1) * (j == 1),
                                    (pn - 1) * (i > 1) + (-1) * (i == 1),
                                    (pn + m) * (j < n) + (-1) * (j == n),
                                    (pn + 1) * (i < m) + (-1) * (i == m)]
        elif self.boundaryType == 'full-periodic':
            for j in (np.array(range(n)) + 1):
                for i in (np.array(range(m)) + 1):
                    pn = (j - 1) * m + i - 1
                    p_adj[pn, :] = [(pn - m) * (j > 1) + (pn + (n-1)*m) * (j == 1),
                                    (pn - 1) * (i > 1) + (pn + m - 1) * (i == 1),
                                    (pn + m) * (j < n) + (pn - (n-1)*m) * (j == n),
                                    (pn + 1) * (i < m) + (pn - (m - 1)) * (i == m)]
        for i in range(nCells):
            self.ngh_pores[i] = p_adj[i][p_adj[i] > -1]
        del p_adj


    def make_pore_tube_connectivity(self):
        """
        Here self.tp_adj, self.ngh_tubes, self.nr_t are set
        tp_adj is a list of all tubes (with size number of tubes/faces) containing lists with size two that
        contain the neighboring cells. [[cell0, cell1], [cell0, cell1], ...]
        ngh_tubes contains the neighboring faces (tubes) for each cell
        nr_t is the total number of tubes (faces)
        :return:
        """
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
                #this dictionary knows all cell-face combinations
                tube_dict[key] = tubeNumber
                tubeNumber += 1
        #all cell pairs that enclose a face
        self.tp_adj = tp_adj[1:, :]
        t_adj = -1*np.ones(nr_p,dtype=np.object)
        for i in range(nr_p):
            nghArray = ngh_pores[i]
            for j in nghArray:
                t_adj[i] = np.append(t_adj[i],tube_dict[tuple(sorted((i,j)))])
        for i in range(nr_p):
            #lists for every cell the according tubes to other cells
            self.ngh_tubes[i] = t_adj[i][t_adj[i] > -1]
        self.nr_t = tubeNumber
        del tube_dict
        del t_adj

    def make_y_faces_array(self):
        """
        :return: yFaces: boolean array size number of faces, an entry is True if a cell is a y-Face
                 (a y-Face is perpendicular to the y-axis)
                 d_cell_numbers: array containing the cell number difference for the neighbor cells of a
                                 face
        """
        m = self.m
        faceCells = self.tp_adj
        d_cell_numbers = abs(faceCells[:, 1] - faceCells[:, 0])
        if self.boundaryType == 'non-periodic':
            yFaces = (d_cell_numbers == 1)
        else:
            d_cell_numbers = np.array(d_cell_numbers)
            yFaces = np.logical_or((d_cell_numbers==1), (d_cell_numbers == m-1))
        return yFaces, d_cell_numbers


    def make_face_list_for_cells(self):
        """
        Here we order the neighboring cells and faces for each cell [left, right, bottom, top]
        if a cell is missing any of these neighbors there is a minus one for that entry
        :return: facelist_array: array of neighboring faces
                 nghlist_array: array of neighboring cells
        """
        nr_cells = self.nr_p
        y_faces, d_cell_numbers = self.y_faces, self.d_cell_numbers
        face_adj_list = self.tp_adj
        m = self.m
        # each cell has four faces 0,1: left and right - 2,3: bottom and top
        facelist_array = -1*np.ones((nr_cells, 4), dtype = np.int)
        nghlist_array = -1*np.ones((nr_cells, 4), dtype = np.int)
        for face in range(self.nr_t):
            #find adjacent cells
            adj_faces = face_adj_list[face]
            cell1, cell2 = adj_faces[0], adj_faces[1]
            if ~y_faces[face]:
                if d_cell_numbers[face] == m:
                    l, r = min(cell1, cell2), max(cell1, cell2)
                else:
                    l, r = max(cell1, cell2), min(cell1, cell2)
                # left face for right cell
                facelist_array[r, 0] = face
                nghlist_array[r, 0] = l
                # right face for left cell
                facelist_array[l, 1] = face
                nghlist_array[l, 1] = r
            else:
                # y face
                if d_cell_numbers[face] == 1:
                    up, dwn = min(cell1, cell2), max(cell1, cell2)
                else:
                    up, dwn = max(cell1, cell2), min(cell1, cell2)
                # lower face for top cell
                facelist_array[up, 2] = face
                nghlist_array[up, 2] = dwn
                # upper face for down cell
                facelist_array[dwn, 3] = face
                nghlist_array[dwn, 3] = up
        return facelist_array, nghlist_array

    def set_periodic_faces(self):
        """
        indicate which faces are periodic (connecting two sides of the domain)
        :return: a set containing the periodic faces
        """
        periodic_faces = set()
        d_cell_numbers = self.d_cell_numbers
        y_faces = self.y_faces
        m = self.m
        for face in range(self.nr_t):
            if y_faces[face]:
                if d_cell_numbers[face] != 1:
                    periodic_faces.add(face)
            else:
                if d_cell_numbers[face] != m:
                    periodic_faces.add(face)
        return periodic_faces

    def get_pore_mask_from_pore_indices(self, pi_list):
        pore_mask = np.zeros(self.nr_p, dtype=np.bool)
        pore_mask[pi_list] = True
        return pore_mask

    def set_transmissibility(self, k, darcy_conversion=False):
        """
        set the transmissibility (face_permeability/dl) for each face
        :param k: permeability field size number of cells (m*n)
        :return: no return value
        """
        if darcy_conversion:
            alpha = 0.001127
        else:
            alpha = 1.0
        nFaces = self.nr_t
        transRockGeometric = self.transmissibility
        m = self.m
        faceCells = self.tp_adj
        dx, dy, dz = self.dx, self.dy, self.dz
        # yFaces between i,j i,j+1
        # ~yFaces between i,j i+1,j
        yFaces, dCellNumbers = self.y_faces, self.d_cell_numbers
        for face in range(nFaces):
            cell1 = faceCells[face, 0]
            cell2 = faceCells[face, 1]
            # choose the correct component of k for that face
            if ~yFaces[face]:
                k1 = k[cell1]
                k2 = k[cell2]
                A = dy * dz
                d = dx
            else:
                k1 = k[cell1]
                k2 = k[cell2]
                A = dx * dz
                d = dy
            transRockGeometric[face] = alpha * (A / d) * 2.0 * k1 * k2 / (k1 + k2)

    def choose_updwn_stream_cell_x(self, d_cell_number, m, adj_cells):
        """
        assuming there is a flow from left to right choose upstream cell, or
        between two adjacent cells choose which one is on the left and which one on the right
        :param d_cell_number: array size number of faces, the absolute difference between ngh cell numbers
        :param m: number of cells in the y direction
        :param adj_cells: list size 2, containing the adjacent cells
        :return: upstream and downstream cell
        """
        if d_cell_number == m:
            ups, dwn = min(adj_cells), max(adj_cells)
        else:
            ups, dwn = max(adj_cells), min(adj_cells)
        return ups, dwn

    def choose_updwn_stream_cell_y(self, d_cell_number, adj_cells):
        """
        assuming there is a flow from bottom to top choose upstream cell, or
        between two adjacent cells choose which one is on the bottom and which one on top
        :param d_cell_number: array size number of faces, the absolute difference between ngh cell numbers
        :param adj_cells: list size 2, containing the adjacent cells
        :return: upstream and downstream cell
        """
        if d_cell_number == 1:
            ups, dwn = max(adj_cells), min(adj_cells)
        else:
            ups, dwn = min(adj_cells), max(adj_cells)
        return ups, dwn

    def make_updwn_cells(self):
        """
        decide for each x-face which cell is on the left and wich cell is on the right
        and for each y-face which cell is on the bottom and which cell is on top
        :return: a list size number of faces each entry is a list of neighboring cells, ordered properly
        """
        y_faces, d_cell_numbers = self.y_faces, self.d_cell_numbers
        face_adj_list = self.tp_adj
        m = self.m
        updwn_cells = []
        for face in range(self.nr_t):
            # find adjacent cells
            adj_cells = face_adj_list[face]
            if ~y_faces[face]:
                ups, dwn = self.choose_updwn_stream_cell_x(d_cell_numbers[face], m, adj_cells)
            else:
                # y face
                ups, dwn = self.choose_updwn_stream_cell_y(d_cell_numbers[face], adj_cells)
            updwn_cells.append([ups, dwn])
        return updwn_cells