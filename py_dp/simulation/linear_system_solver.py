# Copyright 2017 Amir Hossein Delgoshaie, amirdel@stanford.edu
# build on Karim Khayrat's work
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
from scipy.sparse import coo_matrix
from scipy.sparse.linalg.dsolve import linsolve
from scipy.sparse.linalg import gmres
from pyamg import smoothed_aggregation_solver
from scipy.sparse import csc_matrix, linalg as sla
import hashlib
import collections
import functools

sf = 1.0

class BoundedOrderedDict(collections.OrderedDict):
    def __init__(self, *args, **kwds):
        self.maxlen = kwds.pop("maxlen", None)
        collections.OrderedDict.__init__(self, *args, **kwds)
        self._checklen()

    def __setitem__(self, key, value):
        collections.OrderedDict.__setitem__(self, key, value)
        self._checklen()

    def _checklen(self):
        if self.maxlen is not None:
            while len(self) > self.maxlen:
                self.popitem(last=False)


def memoize(func=None, maxlen=None):
    if func:
        cache = BoundedOrderedDict(maxlen=maxlen)
        @functools.wraps(func)
        def memo_target(arg):
            b = arg.data.view(np.uint8)
            hashed = hashlib.sha1(b).hexdigest()
            lookup_value = hashed
            if lookup_value not in cache:
                cache[lookup_value] = func(arg)
            return cache[lookup_value]
        return memo_target
    else:
        def memoize_factory(func):
            return memoize(func, maxlen=maxlen)
        return memoize_factory


@memoize(maxlen=10)
def get_lu_factorization_memoized(A):
    lu = sla.splu(A)
    return lu


class ConductanceMatrixStandard(object):
    """
    To understand the following code, one needs to know about the COO sparse matrix format and numpy
    """

    def __init__(self, network):
        self.network = network

        self.coo_len = 2 * network.nr_t + network.nr_p
        self.data = np.zeros(self.coo_len)

        self.row, self.col = self.get_row_and_col_arrays()

        self.data_tube_ind = np.ones(2 * network.nr_t, dtype=np.int32)
        self.data_tube_ind[:] = np.hstack(network.ngh_tubes.flat)

        self.create_index_lists()
        self.N = self.network.nr_p

    def get_row_and_col_arrays(self):
        #Storage format: COL: [N_1_1 N_1_2 N_1_3 N_1_4 P_1 N_2_1 N_2_2 N_2_3 N_2_4 P_2 ...]
        #                ROW: [P_1   P_1   P_1   P_1   P_1 P_2   P_2   P_2   P_2   P_2 ...]

        #N_1_3:  3rd Neighbouring pore of pore #1 , P_5: Pore #5
        #P_1, P_2... always in increasing order.

        row = -np.ones(self.coo_len, dtype=np.int32)
        col = -np.ones(self.coo_len, dtype=np.int32)

        network = self.network
        start = 0

        for p_i in xrange(network.nr_p):
            col_entries = network.ngh_pores[p_i]
            end = start + network.nr_nghs[p_i]

            row[start:end] = p_i
            col[start:end] = col_entries
            row[end] = p_i
            col[end] = p_i

            start = end + 1

        assert (len(row[row < 0]) == 0)
        assert (len(col[col < 0]) == 0)

        return row, col

    def create_index_lists(self):
        self.data_diag_mask = (self.row == self.col)
        self.data_nondiag_mask = (self.row != self.col)
        self.data_diag_ind = self.data_diag_mask.nonzero()[0]
        self.data_nondiag_ind = self.data_nondiag_mask.nonzero()[0]
        assert (np.sum(self.data_nondiag_mask) == len(self.data_tube_ind))

    def fill_with_conductances(self, conductivity):
        network = self.network
        nr_p = network.nr_p
        self.data[:] = 0.0

        #Set non-diagonal entries
        self.data[self.data_nondiag_ind] = -conductivity[self.data_tube_ind]

        #Set diagonal entries
        A = coo_matrix((self.data, (self.row, self.col)), shape=(nr_p, nr_p))
        self.data[self.data_diag_ind] = -A * np.ones(network.nr_p)

        #assert (np.isclose(np.sum(self.data), 0.0, atol=1e-15))

    def set_dirichlet_pores(self, pi_bnd):
        pore_mask = self.network.get_pore_mask_from_pore_indices(pi_bnd)
        data_diag_mask_bnd = (pore_mask[self.row]) & (pore_mask[self.col]) & self.data_diag_mask
        data_nondiag_mask_bnd = (pore_mask[self.row]) & np.logical_not(data_diag_mask_bnd)
        self.data[data_diag_mask_bnd] = 1.0 * sf
        self.data[data_nondiag_mask_bnd] = 0.0
        self.fix_isolated_pores()

    def fix_isolated_pores(self):
        data_diag_mask = np.zeros(self.coo_len, dtype=np.bool)
        data_diag_mask[self.data_diag_ind] = True
        self.data[data_diag_mask & (self.data == 0.0)] = 1.0 * sf

    def get_csr_matrix(self):
        network = self.network
        nr_p = network.nr_p
        return coo_matrix((self.data, (self.row, self.col)), shape=(nr_p, nr_p)).tocsr()

    def get_csc_matrix(self):
        network = self.network
        nr_p = network.nr_p
        return coo_matrix((self.data, (self.row, self.col)), shape=(nr_p, nr_p)).tocsc()

    def get_coo_matrix(self):
        return self.row, self.col, self.data

    def add_to_diagonals(self, val):
        self.data[self.data_diag_ind] += val

    def set_dirichlet_inlet(self):
        self.set_dirichlet_pores(self.network.pi_in)

    def set_dirichlet_outlet(self):
        self.set_dirichlet_pores(self.network.pi_out)

    def add_const_to_inlet_diagonals(self, val):
        pi_in_mask = self.network.get_pore_mask_from_pore_indices(self.network.pi_in)
        data_diag_ind_inlet = (pi_in_mask[self.row]) & (pi_in_mask[self.col]) & (self.data_diag_mask)
        self.data[data_diag_ind_inlet] += val

    def set_diagonals_to_zero(self):
        #Reconstruct data on all diagonal entries
        self.data[self.data_diag_ind] = 0.0

    def remove_connections(self, source_color, target_color, color_map):
        source_mask = color_map[self.row] == source_color
        target_mask = color_map[self.col] == target_color
        data_mask = target_mask & source_mask

        #Remove data from off-diagonal entries
        self.data[data_mask] = 0.0

        #Reconstruct data on all diagonal entries
        self.data[self.data_diag_ind] = 0.0
        #Set diagonal entries
        A = coo_matrix((self.data, (self.row, self.col)), shape=(self.N, self.N))
        self.data[self.data_diag_ind] = -A * np.ones(self.network.nr_p)
        assert (np.isclose(np.sum(self.data), 0.0, atol=1e-15))


class ConductanceMatrixExtended(object):
    def set_dirichlet_outlet(self):
        self.standard_matrix.set_dirichlet_outlet()

    def fill_with_conductances(self, conductivity):
        self.standard_matrix.fill_with_conductances(conductivity)

    def get_csr_matrix(self):
        row_s, col_s, data_s = self.standard_matrix.get_coo_matrix()

        row = np.hstack((row_s, self.row))
        col = np.hstack((col_s, self.col))
        data = np.hstack((data_s, self.data))
        return coo_matrix((data, (row, col)), shape=(self.N, self.N)).tocsr()

    def get_csc_matrix(self):
        row_s, col_s, data_s = self.standard_matrix.get_coo_matrix()

        row = np.hstack((row_s, self.row))
        col = np.hstack((col_s, self.col))
        data = np.hstack((data_s, self.data))
        return coo_matrix((data, (row, col)), shape=(self.N, self.N)).tocsc()


class RHSStandard(object):
    def __init__(self, network):
        self.val = np.zeros(network.nr_p)
        self.network = network

    def set_dirichlet_pores(self, pi_list, value):
        self.val[pi_list] = value * sf

    def set_neumann_pores_distributed(self, pi_list, value_list):
        self.val[pi_list] = value_list

    def set_dirichlet_inlet(self, value):
        self.set_dirichlet_pores(self.network.pi_in, value)

    def set_dirichlet_outlet(self, value):
        self.set_dirichlet_pores(self.network.pi_out, value)


class LinearSystem(object):
    def fill_matrix(self, conductances):
        self.matrix.fill_with_conductances(conductances)

    def solve(self, solver="AMG"):
        A = self.matrix.get_csr_matrix()
        if(solver == "AMG"):
            ml = smoothed_aggregation_solver(A)
            M = ml.aspreconditioner()
            self.sol = gmres(A, self.rhs.val, M=M)[0]
        elif( solver == "LU"):
            lu = get_lu_factorization_memoized(A)
            self.sol = lu.solve(self.rhs.val)

    def solve_with_initial_guess(self, initial_guess, solver="LU"):
        A = self.matrix.get_csr_matrix()

        if(solver == "AMG"):
            ml = smoothed_aggregation_solver(A)
            M = ml.aspreconditioner()
            self.sol = gmres(A, self.rhs.val, x0=initial_guess, M=M)[0]
        elif( solver == "LU"):
            self.sol = linsolve.spsolve(A, self.rhs.val)


    def get_pressure(self):
        return self.sol[0:self.network.nr_p]


class LinearSystemStandard(LinearSystem):
    def __init__(self, network):
        self.network = network
        self.matrix = ConductanceMatrixStandard(network)
        self.rhs = RHSStandard(network)
        self.sol = None

    def set_dirichlet_pores(self, pi_list, value):
        self.matrix.set_dirichlet_pores(pi_list)
        self.rhs.set_dirichlet_pores(pi_list, value)

    def set_neumann_pores(self, pi_list, value_list):
        self.rhs.set_neumann_pores_distributed(pi_list, -value_list)

    def set_neumann_pores_equally_divided(self, pi_list, value):
        nr_pores = len(pi_list)
        new_value = value/nr_pores
        value_list = -1*np.ones(nr_pores)*new_value
        self.rhs.set_neumann_pores_distributed(pi_list, value_list)

    def set_neumann_pores_vol_distributed(self, pi_list, value):
        volume_list = self.network.pores.vol[pi_list]
        sum_vol = np.sum(volume_list)
        ratio_list = volume_list/sum_vol
        value_list = -1*value*ratio_list
        self.rhs.set_neumann_pores_distributed(pi_list, value_list)

    def set_dirichlet_inlet(self, value):
        self.set_dirichlet_pores(self.network.pi_in, value)

    def set_dirichlet_outlet(self, value):
        self.set_dirichlet_pores(self.network.pi_out, value)

    def remove_connections(self, source_color, target_color, color_map):
        self.matrix.remove_connections(source_color, target_color, color_map)


class LSGridPeriodicPurturbations(LinearSystemStandard):
    """
    Solver class used for solving steady state problem on periodic grids and setting mean
    pressure gradient in the x and y direction. The linear system is set for the fluctuations around the
    value.
    """
    def __init__(self, network):
        super(LSGridPeriodicPurturbations, self).__init__(network)

    def periodic_rhs_vec(self, dp_x, dp_y):
        # TODO: needs to have at least three rows and columns, no problem for my case
        """
        find the rhs for the full periodic case
        :param dp_x: average pressure difference of the left and right boundary (P_l - P_r)
        :param dp_y: average pressure difference of the bottom and top boundary (P_b - P_t)
        :return: rhs vector n_cells (nr_p)
        """
        grid = self.network
        lx, ly = grid.lx, grid.ly
        rhs_vec = np.zeros(grid.nr_p)
        nFaces = grid.nr_t
        transRockGeometric = grid.transmissibility
        faceCells = grid.updwn_cells
        dx, dy, dz = grid.dx, grid.dy, grid.dz
        dCellNumbers, yFaces = grid.d_cell_numbers, grid.y_faces
        for face in range(nFaces):
            adj_cells = faceCells[face]
            trans = transRockGeometric[face]
            ups, dwn = adj_cells[0], adj_cells[1]
            # choose the correct component of dp for that face
            if ~yFaces[face]:
                d, dp, l = dx, dp_x, lx
            else:
                # y face
                d, dp, l = dy, dp_y, ly
            rhs_vec[ups] -= (dp/l)*trans*d
            rhs_vec[dwn] += (dp/l)*trans*d
        return rhs_vec

    def set_face_velocity(self, dp_x, dp_y):
        """
        function to set the right hand side vector when solving for pressure fluctuations
        :param dp_x: average pressure difference of the left and right boundary (P_l - P_r)
        :param dp_y: verage pressure difference of the bottom and top boundary (P_b - P_t)
        :return: velocity of at the cell faces (grid.nr_t)
        """
        grid = self.network
        p_fluc = self.sol
        lx, ly = grid.lx, grid.ly
        dx, dy, dz = grid.dx, grid.dy, grid.dz
        y_faces, d_cell_numbers = grid.y_faces, grid.d_cell_numbers
        face_adj_list = grid.updwn_cells
        transRockGeometric = grid.transmissibility
        face_velocity = np.zeros(grid.nr_t)
        for face in range(grid.nr_t):
            # find adjacent cells
            adj_cells = face_adj_list[face]
            trans = transRockGeometric[face]
            ups, dwn = adj_cells[0], adj_cells[1]
            if ~y_faces[face]:
                d, dp, l = dx, dp_x, lx
            else:
                # y face
                d, dp, l = dy, dp_y, ly
            A = dz*d
            face_velocity[face] = trans*(d*dp/l + (p_fluc[ups]-p_fluc[dwn]))/A
        return face_velocity

    def get_cell_velocity(self):
        grid = self.network
        face_velocities = grid.face_velocities
        cell_faces = grid.facelist_array
        u = 0.5*(face_velocities[cell_faces[:,0]] + face_velocities[cell_faces[:,1]])
        v = 0.5 * (face_velocities[cell_faces[:, 2]] + face_velocities[cell_faces[:, 3]])
        return u,v


class LSGridPressure(LinearSystemStandard):
    """
    a class for solving the steady state on phase pressure equation
    """

    def get_face_velocity(self):
        grid = self.network
        p = self.sol
        lx, ly = grid.lx, grid.ly
        dx, dy, dz = grid.dx, grid.dy, grid.dz
        y_faces, d_cell_numbers = grid.y_faces, grid.d_cell_numbers
        face_adj_list = grid.updwn_cells
        transRockGeometric = grid.transmissibility
        face_velocity = np.zeros(grid.nr_t)
        for face in range(grid.nr_t):
            # find adjacent cells
            adj_cells = face_adj_list[face]
            trans = transRockGeometric[face]
            ups, dwn = adj_cells[0], adj_cells[1]
            if ~y_faces[face]:
                d, l = dx, lx
            else:
                # y face
                d, l = dy, ly
            A = dz * d
            face_velocity[face] = trans * (p[ups] - p[dwn]) / A
        return face_velocity