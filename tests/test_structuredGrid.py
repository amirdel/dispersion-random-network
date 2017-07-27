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

#testing
from unittest import TestCase, skip
from py_dp.simulation.grid_structured import structuredGrid
from numpy import array
import numpy as np
import pickle
import os

class TestStructuredGrid(TestCase):
    def test_make_pore_connectivity_list(self):
        grid = structuredGrid(4, 3, 1, 1)
        # test the connectivity of the cells
        expected_connectivity = [array([4, 1]), array([0, 5, 2]), array([1, 6, 3]), array([2, 7]),
                                 array([0, 8, 5]), array([1, 4, 9, 6]), array([2, 5, 10, 7]),
                                 array([3, 6, 11]), array([4, 9]), array([5, 8, 10]), array([6, 9, 11]),
                                 array([7, 10])]
        output_connectivity = grid.ngh_pores
        for i in range(len(output_connectivity)):
            assert (np.all(expected_connectivity[i] == output_connectivity[i]))
        # test the number of neighbors array
        expected_nr_ngh = [2, 3, 3, 2, 3, 4, 4, 3, 2, 3, 3, 2]
        np.testing.assert_equal(expected_nr_ngh, grid.nr_nghs)
        expected_tp_adj = [[0, 4], [0, 1], [1, 5], [1, 2], [2, 6], [2, 3], [3, 7],
                           [4, 8], [4, 5], [5, 9], [5, 6], [6, 10], [6, 7], [7, 11], [8, 9],
                           [9, 10], [10, 11]]
        np.testing.assert_equal(grid.tp_adj, expected_tp_adj)

        grid = structuredGrid(4, 3, 1, 1, boundaryType='full-periodic')
        # test the connectivity of the cells
        expected_connectivity = [array([8, 3, 4, 1]), array([9, 0, 5, 2]), array([10, 1, 6, 3]), array([11, 2, 7, 0]),
                                 array([0, 7, 8, 5]), array([1, 4, 9, 6]), array([2, 5, 10, 7]),
                                 array([3, 6, 11, 4]), array([4, 11, 0, 9]), array([5, 8, 1, 10]), array([6, 9, 2, 11]),
                                 array([7, 10, 3, 8])]
        output_connectivity = grid.ngh_pores
        for i in range(len(output_connectivity)):
            np.testing.assert_equal(expected_connectivity[i], output_connectivity[i])
        # test the number of neighbors array
        expected_nr_ngh = [4] * grid.nr_p
        np.testing.assert_equal(expected_nr_ngh, grid.nr_nghs)

    def test_tube_adj_cells(self):
        grid = structuredGrid(4, 3, 1, 1)
        expected_tp_adj = [[0, 4], [0, 1], [1, 5], [1, 2], [2, 6], [2, 3], [3, 7],
                           [4, 8], [4, 5], [5, 9], [5, 6], [6, 10], [6, 7], [7, 11], [8, 9],
                           [9, 10], [10, 11]]
        np.testing.assert_equal(grid.tp_adj, expected_tp_adj)
        grid = structuredGrid(4, 3, 1, 1, boundaryType='full-periodic')
        print grid.tp_adj
        expected_tp_adj = [[0, 8], [0, 3], [0, 4], [0, 1], [1, 9], [1, 5], [1, 2],
                           [2, 10], [2, 6], [2, 3], [3, 11], [3, 7], [4, 7], [4, 8],
                           [4, 5], [5, 9], [5, 6], [6, 10], [6, 7], [7, 11], [8, 11],
                           [8, 9], [9, 10], [10, 11]]
        np.testing.assert_equal(grid.tp_adj, expected_tp_adj)

    def test_set_transmissibility(self):
        grid = structuredGrid(4, 3, 1, 1)
        perm = [2, 2, 2, 2, 6, 6, 6, 6, 2, 2, 2, 2]
        grid.set_transmissibility(perm)
        trans = grid.transmissibility
        for i in range(grid.nr_t):
            tp = grid.tp_adj[i]
            c1, c2 = tp[0], tp[1]
            expected_trans = 2*perm[c1]*perm[c2]/(perm[c1]+perm[c2])
            self.assertEqual(expected_trans, trans[i])
        grid = structuredGrid(4, 3, 1, 1, boundaryType='full-periodic')
        perm = [2, 2, 2, 2, 6, 6, 6, 6, 2, 2, 2, 2]
        grid.set_transmissibility(perm)
        trans = grid.transmissibility
        for i in range(grid.nr_t):
            tp = grid.tp_adj[i]
            c1, c2 = tp[0], tp[1]
            expected_trans = 2 * perm[c1] * perm[c2] / (perm[c1] + perm[c2])
            self.assertEqual(expected_trans, trans[i])

    def test_make_y_faces(self):
        grid = structuredGrid(4, 3, 1, 1)
        expected_y_faces = [False, True, False, True, False, True, False,
                            False, True, False, True, False, True, False,
                            True,  True,  True]
        expected_d_cell_numbers = [4,1, 4, 1, 4, 1, 4, 4, 1, 4, 1,
                                   4, 1, 4, 1, 1, 1]
        np.testing.assert_equal(expected_y_faces, grid.y_faces)
        np.testing.assert_equal(expected_d_cell_numbers, grid.d_cell_numbers)
        grid = structuredGrid(4, 3, 1, 1, boundaryType='full-periodic')
        expected_y_faces = [False, True, False, True, False, False, True,
                            False, False, True, False, False, True, False,
                            True, False, True, False, True, False, True, True,
                            True, True]
        expected_d_cell_numbers = [8, 3, 4, 1, 8, 4, 1, 8, 4, 1, 8, 4, 3, 4,
                                   1, 4, 1, 4, 1, 4, 3, 1, 1, 1]
        np.testing.assert_equal(expected_y_faces, grid.y_faces)
        np.testing.assert_equal(expected_d_cell_numbers, grid.d_cell_numbers)

    def test_face_list_for_cells(self):
        grid = structuredGrid(4, 3, 1, 1)
        expected_facelist_array = [[-1 ,0 ,1 ,-1], [-1 ,2 ,3 ,1], [-1 ,4 ,5 ,3], [-1 ,6 ,-1 ,5],
                                   [0 ,7 ,8 ,-1], [2 ,9 ,10 ,8], [4 ,11 ,12 ,10], [6 ,13 ,-1 ,12],
                                   [7 ,-1 ,14 ,-1], [9 ,-1 ,15 ,14], [11 ,-1 ,16 ,15], [13 ,-1 ,-1 ,16]]
        self.assertNestedEqual(expected_facelist_array, grid.facelist_array)
        expected_nghlist_array = [[-1 ,4 ,1 ,-1], [-1 ,5 ,2 ,0], [-1 ,6 ,3 ,1], [-1 ,7 ,-1 ,2], [0 ,8 ,5 ,-1],
                                  [1 ,9 ,6 ,4], [2 ,10 ,7 ,5], [3 ,11 ,-1 ,6], [4 ,-1 ,9 ,-1], [5 ,-1 ,10 ,8],
                                  [6 ,-1 ,11 ,9], [7 ,-1 ,-1 ,10]]
        self.assertNestedEqual(expected_nghlist_array, grid.nghlist_array)
        grid = structuredGrid(4, 3, 1, 1, boundaryType='full-periodic')
        expected_facelist_array = [[0 ,2 ,3 ,1], [4 ,5 ,6 ,3], [7 ,8 ,9 ,6], [10 ,11 ,1 ,9],
                                   [2 ,13 ,14 ,12], [5 ,15 ,16 ,14], [8 ,17 ,18 ,16], [11 ,19 ,12 ,18],
                                   [13 ,0 ,21 ,20], [15 ,4 ,22 ,21], [17 ,7 ,23 ,22], [19 ,10 ,20 ,23]]
        self.assertNestedEqual(expected_facelist_array, grid.facelist_array)
        expected_nghlist_array = [[8 ,4 ,1 ,3], [9 ,5 ,2 ,0], [10 ,6 ,3 ,1], [11 ,7 ,0 ,2], [0 ,8 ,5 ,7],
                                  [1 ,9 ,6 ,4], [2 ,10 ,7 ,5], [3 ,11 ,4 ,6], [4 ,0 ,9 ,11], [5 ,1 ,10 ,8],
                                  [6 ,2 ,11 ,9], [7 ,3 ,8 ,10]]
        self.assertNestedEqual(expected_nghlist_array, grid.nghlist_array)

    def print_nested_list(self, nested_list):
        str_array = []
        for idx, item in enumerate(nested_list):
            if idx > 0:
                prefix = ', '
            else:
                prefix = ''
            tmp = [str(item[i]) + ' ,' for i in range(len(item))]
            tmp[-1] = tmp[-1][:-2]
            str_array.append(prefix + '[' + ''.join(tmp) + ']')
        print '[' + ''.join(str_array) + ']'

    def assertNestedEqual(self, list1, list2):
        assert(len(list1) == len(list2))
        for i in range(len(list1)):
            np.testing.assert_equal(list1[i], list2[i])

    def test_choose_updwn_cells(self):
        grid = structuredGrid(4, 3, 1, 1, boundaryType='full-periodic')
        expected_updwn = [[8 ,0], [0 ,3], [0 ,4], [1 ,0], [9 ,1], [1 ,5], [2 ,1], [10 ,2], [2 ,6],
                          [3 ,2], [11 ,3], [3 ,7], [4 ,7], [4 ,8], [5 ,4], [5 ,9], [6 ,5], [6 ,10],
                          [7 ,6], [7 ,11], [8 ,11], [9 ,8], [10 ,9], [11 ,10]]
        for i in range(len(grid.updwn_cells)):
            np.testing.assert_equal(expected_updwn[i], grid.updwn_cells[i])

    # def test_periodic_rhs_vec(self):
    #     # this_dir =  os.path.dirname(os.path.dirname(__file__))
    #     # grid_save_folder = os.path.join(this_dir, 'test_related_files/grids')
    #     # grid_name = '4_3_periodic.pkl'
    #     # grid_save_path = os.path.join(grid_save_folder, grid_name)
    #     # with open(grid_save_path, 'r') as input:
    #     #     grid = pickle.load(input)
    #     grid = structuredGrid(4, 3, 1, 1, boundaryType='full-periodic')
    #     dp_x = 0.0
    #     dp_y = 100.0
    #     perm = np.ones(grid.nr_p)
    #     grid.set_transmissibility(perm)
    #     rhs_vec = grid.periodic_rhs_vec(dp_x, dp_y)
    #     np.testing.assert_equal(rhs_vec, np.zeros(grid.nr_p))
    #     dp_x = 100.0
    #     dp_y = 0.0
    #     rhs_vec = grid.periodic_rhs_vec(dp_x, dp_y)
    #     np.testing.assert_equal(rhs_vec, np.zeros(grid.nr_p))

    def test_boundary_faces(self):
        grid = structuredGrid(4, 3, 1, 1, boundaryType='full-periodic')
        expected_periodic_faces = set([0, 1, 4, 7, 10, 12, 20])
        self.assertSetEqual(expected_periodic_faces, grid.periodic_faces)

    def test_dummytest(self):
        pass