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

from unittest import TestCase
import numpy as np
from py_dp.simulation.grid_structured import structuredGrid
from py_dp.dispersion.dispersion_continua import dispersionSystemContinua

class TestDispersionSystemContinua(TestCase):
    def test_find_exit_conditions_1d(self):
        # case1: v1<v2 and both are positive
        grid = structuredGrid(4, 3, 1.0, 1.0, boundaryType='full-periodic')
        dx, x1, x2 = 1.0, 0.0, 1.0
        f1, f2 = 0,1
        ds1 = dispersionSystemContinua(grid, 1, 1, tracking_type='exit')
        x = 0.0
        v1 = 10.0
        v2 = 12.0
        exit, exit_idx, xe, v, ve, dt = ds1.find_exit_conditions_1d(dx, x, x1, f1, v1, x2, f2, v2)
        self.assertListEqual([exit, exit_idx, xe, v, ve] , [True, f2, x2, v1, v2])
        self.assertAlmostEqual(dt, 0.0911, places=3)
        # case2: v1>v2 and both are negative
        x = 1.0
        v1 = -1.0
        v2 = -2.0
        exit, exit_idx, xe, v, ve, dt = ds1.find_exit_conditions_1d(dx, x, x1, f1, v1, x2, f2, v2)
        self.assertListEqual([exit, exit_idx, xe, v, ve], [True, f1, x1, v2, v1])
        self.assertAlmostEqual(dt, 0.693, places=3)
        # case3: v1==v2 and both are negative
        x = 0.5
        v1 = -2.0
        v2 = v1
        exit, exit_idx, xe, v, ve, dt = ds1.find_exit_conditions_1d(dx, x, x1, f1, v1, x2, f2, v2)
        self.assertListEqual([exit, exit_idx, xe, v, ve], [True, f1, x1, v2, v2])
        self.assertEqual(dt, 0.25)
        # case4: v1==v2 and both are positive
        x = 0.5
        v1 = 2.0
        v2 = v1
        exit, exit_idx, xe, v, ve, dt = ds1.find_exit_conditions_1d(dx, x, x1, f1, v1, x2, f2, v2)
        self.assertListEqual([exit, exit_idx, xe, v, ve], [True, f2, x2, v2, v2])
        self.assertEqual(dt, 0.25)
        # case5: v1>0 v2<0 no way to get out
        x = 0.5
        v1 = 2.0
        v2 = -1.0
        exit, exit_idx, xe, v, ve, dt = ds1.find_exit_conditions_1d(dx, x, x1, f1, v1, x2, f2, v2)
        self.assertEqual(exit, False)
        # case5: v1<0 and v2>0, on the right side of the stagnation plane
        x = 0.7
        v1 = -1.0
        v2 = 1.0
        exit, exit_idx, xe, v, ve, dt = ds1.find_exit_conditions_1d(dx, x, x1, f1, v1, x2, f2, v2)
        self.assertListEqual([exit, exit_idx, xe, ve], [True, f2, x2, v2])
        self.assertAlmostEqual(dt, 0.458145, places=4)
        # case6: v1<0 and v2>0, on the left side of the stagnation plane
        x = 0.3
        v1 = -1.0
        v2 = 1.0
        exit, exit_idx, xe, v, ve, dt = ds1.find_exit_conditions_1d(dx, x, x1, f1, v1, x2, f2, v2)
        self.assertListEqual([exit, exit_idx, xe, ve], [True, f1, x1, v1])
        self.assertAlmostEqual(dt, 0.458145, places=4)
        # case7: v1<0 and v2>0, on the stagnation plane
        x = 0.5
        v1 = -1.0
        v2 = 1.0
        exit, exit_idx, xe, v, ve, dt = ds1.find_exit_conditions_1d(dx, x, x1, f1, v1, x2, f2, v2)
        self.assertEqual(exit, False)
        # case8: v1=0 and v2>0
        x = 0.5
        v1 = 0.0
        v2 = 1.0
        exit, exit_idx, xe, v, ve, dt = ds1.find_exit_conditions_1d(dx, x, x1, f1, v1, x2, f2, v2)
        self.assertListEqual([exit, exit_idx, xe, ve], [True, f2, x2, v2])
        # case9: v2=0 and v1<0
        x = 0.5
        v1 = -1.0
        v2 = 0.0
        exit, exit_idx, xe, v, ve, dt = ds1.find_exit_conditions_1d(dx, x, x1, f1, v1, x2, f2, v2)
        self.assertListEqual([exit, exit_idx, xe, ve], [True, f1, x1, v1])
        # print exit, exit_idx, xe, v, ve, dt
        # self.fail()



    def test_find_exit_conditions(self):
        grid = structuredGrid(4, 3, 1.0, 1.0, boundaryType='full-periodic')
        # set face velocities for grid
        grid.face_velocities = np.zeros(grid.nr_t)
        ngh_faces = [0, 2, 3, 1]
        ngh_cells = [8, 4, 1, 3]
        vl, vr = 4.0, 4.0
        vb, vt = 1.0, 1.0
        grid.face_velocities[ngh_faces] = [vl, vr, vb, vt]
        # starting position center of left face
        xs = grid.pores.x[0] - grid.dx/2
        ys = grid.pores.y[0]
        ds1 = dispersionSystemContinua(grid, 1, 1, tracking_type='exit')
        exit_cell, exit_face, xe, ye, te = ds1.find_exit_conditions(0, xs, ys, 0.0)
        self.assertListEqual([exit_cell, xe, ye, te], [4, 1.0, 3.75, 0.25])
        # TODO: more test cases for here

    def test_init_particles_left_boundary(self):
        grid = structuredGrid(4, 3, 1.0, 1.0, boundaryType='full-periodic')
        n_particles = 5
        n_steps = 3
        ds1 = dispersionSystemContinua(grid, n_particles, n_steps, tracking_type='exit')
        ds1.init_particles_left_boundary()
        expected_x = [0.5, 0.5, 0.5, 0.5, 1.5]
        expected_y = [3.5, 2.5, 1.5, 0.5, 3.5]
        self.assertListEqual(list(ds1.cell_nr_array[:, 0]), range(n_particles))
        self.assertListEqual(list(ds1.x_array[:, 0]), expected_x)
        self.assertListEqual(list(ds1.y_array[:, 0]), expected_y)

    def test_init_particles_left_buffered(self):
        grid = structuredGrid(4, 3, 1.0, 1.0, boundaryType='full-periodic')
        n_particles = 2
        n_steps = 3
        ds1 = dispersionSystemContinua(grid, n_particles, n_steps, tracking_type='exit')
        ds1.init_particles_left_buffered(1)
        self.assertListEqual(list(ds1.cell_nr_array[:, 0]), [1,2])
        expected_x = [0.5, 0.5]
        expected_y = [2.5, 1.5]
        self.assertListEqual(list(ds1.x_array[:, 0]), expected_x)
        self.assertListEqual(list(ds1.y_array[:, 0]), expected_y)



            # def test_integrate_path(self):
    #     self.fail()
    #
    # def test_follow_all_particles(self):
    #     self.fail()
    #
    # def test_follow_all_particles_exit(self):
    #     self.fail()
    #
    # def test_follow_all_particles_dt(self):
    #     self.fail()
