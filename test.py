# %%
import sys
import numpy as np
# %%
# from MD_functions import EnergyMinimization
from PlotFunctions import ConfigPlot

rand_id = int(sys.argv[1]) # random number seed
N_cell = int(sys.argv[2]) # number of prisms per row
N_row = int(sys.argv[3]) # number of rows of prisms
# %%
KA = 1.0 # spring constant for area energy term
KV = 0.1 # spring constant for volume energy term
R1 = 1.0 # radius of inner ring
R2 = 2.0 # radius of outer ring

# N_cell = 4
# N_row = 3

Nv_cell = 2 * N_cell # number of vertices for all trapezoids per row
Nv_temp = (N_row + 1) * Nv_cell # total number of vertices before adding vertices in the center of each trapezoid and rectangle
Nf = 4 * (4 * N_row + 1) * N_cell # number of triangles after adding vertices at the center of each trapezoid and rectangle

# L1, L2, and L3: lengths for one trapezoid
L1 = R1 * 2.0 * np.sin(np.pi / N_cell)
L2 = R2 * 2.0 * np.sin(np.pi / N_cell)
L3 = R2 - R1
theta = np.arange(N_cell) / N_cell * 2 * np.pi # evenly space vertices for the trapezoids on the inner and outer rings

# indices for next (ift_temp) and previous (jft_temp) vertices along a ring, in a ciruclar way
ift_temp = np.concatenate((np.arange(1, N_cell), np.arange(1)))
jft_temp = np.concatenate((np.arange(N_cell - 1, N_cell), np.arange(N_cell - 1)))

# x1, y1: x and y coordinates of trapezoid vertices on the inner ring
# x2, y2: x and y coordinates of trapezoid vertices on the outer ring
x1 = R1 * np.cos(theta)
y1 = R1 * np.sin(theta)
x2 = R2 * np.cos(theta)
y2 = R2 * np.sin(theta)

xv_cell = np.concatenate((x1, x2)) # x coordinates for all vertices in one row
yv_cell = np.concatenate((y1, y2)) # y coordinates for all vertices in one row
xv_all = np.tile(xv_cell, N_row + 1) # x coordinates for all vertices in all rows
yv_all = np.tile(yv_cell, N_row + 1) # y coordinates for all vertices in all rows
zv_all = np.repeat(np.arange(N_row + 1) * L1, Nv_cell) # z coordinates for all vertices in all rows

f_unit = np.empty((0, 3), dtype = 'int16') # list of vertex indices for all triangles, will be Nf x 3 array
# trapezoid at each row for N_row + 1 rows, add one vertex per each trapezoid
for nr in np.arange(N_row + 1):
    v_offset = nr * 2 * N_cell
    for nc in np.arange(N_cell):
        Nv_temp = Nv_temp + 1
        v1 = nc + v_offset
        v2 = ift_temp[nc] + v_offset
        v3 = v2 + N_cell
        v4 = v1 + N_cell
        f_unit = np.concatenate((f_unit, np.array([[v1, v2, Nv_temp - 1], [v2, v3, Nv_temp - 1], [v3, v4, Nv_temp - 1], [v4, v1, Nv_temp - 1]])))
        xv_temp = np.mean(xv_all[[v1, v2, v3, v4]])
        yv_temp = np.mean(yv_all[[v1, v2, v3, v4]])
        zv_temp = nr * L1
        xv_all = np.concatenate((xv_all, np.array([xv_temp])))
        yv_all = np.concatenate((yv_all, np.array([yv_temp])))
        zv_all = np.concatenate((zv_all, np.array([zv_temp])))

# rectangles on the inner side of the sleeve
for nr in np.arange(N_row):
    v_offset = nr * 2 * N_cell
    for nc in np.arange(N_cell):
        Nv_temp = Nv_temp + 1
        v1 = nc + v_offset
        v2 = ift_temp[nc] + v_offset
        v3 = v2 + 2 * N_cell
        v4 = v1 + 2 * N_cell
        f_unit = np.concatenate((f_unit, np.array([[v1, v2, Nv_temp - 1], [v2, v3, Nv_temp - 1], [v3, v4, Nv_temp - 1], [v4, v1, Nv_temp - 1]])))
        xv_temp = np.mean(xv_all[[v1, v2, v3, v4]])
        yv_temp = np.mean(yv_all[[v1, v2, v3, v4]])
        zv_temp = np.mean(zv_all[[v1, v2, v3, v4]])
        xv_all = np.concatenate((xv_all, np.array([xv_temp])))
        yv_all = np.concatenate((yv_all, np.array([yv_temp])))
        zv_all = np.concatenate((zv_all, np.array([zv_temp])))

# rectangles on the outer side of the sleeve
for nr in np.arange(N_row):
    v_offset = nr * 2 * N_cell
    for nc in np.arange(N_cell):
        Nv_temp = Nv_temp + 1
        v1 = nc + N_cell + v_offset
        v2 = ift_temp[nc] + N_cell + v_offset
        v3 = v2 + 2 * N_cell
        v4 = v1 + 2 * N_cell
        f_unit = np.concatenate((f_unit, np.array([[v1, v2, Nv_temp - 1], [v2, v3, Nv_temp - 1], [v3, v4, Nv_temp - 1], [v4, v1, Nv_temp - 1]])))
        xv_temp = np.mean(xv_all[[v1, v2, v3, v4]])
        yv_temp = np.mean(yv_all[[v1, v2, v3, v4]])
        zv_temp = np.mean(zv_all[[v1, v2, v3, v4]])
        xv_all = np.concatenate((xv_all, np.array([xv_temp])))
        yv_all = np.concatenate((yv_all, np.array([yv_temp])))
        zv_all = np.concatenate((zv_all, np.array([zv_temp])))

# vertical rectangles shared between prisms
for nr in np.arange(N_row):
    v_offset = nr * 2 * N_cell
    for nc in np.arange(N_cell):
        Nv_temp = Nv_temp + 1
        v1 = nc + v_offset
        v2 = v1 + N_cell
        v3 = v2 + 2 * N_cell
        v4 = v1 + 2 * N_cell
        f_unit = np.concatenate((f_unit, np.array([[v1, v2, Nv_temp - 1], [v2, v3, Nv_temp - 1], [v3, v4, Nv_temp - 1], [v4, v1, Nv_temp - 1]])))
        xv_temp = np.mean(xv_all[[v1, v2, v3, v4]])
        yv_temp = np.mean(yv_all[[v1, v2, v3, v4]])
        zv_temp = np.mean(zv_all[[v1, v2, v3, v4]])
        xv_all = np.concatenate((xv_all, np.array([xv_temp])))
        yv_all = np.concatenate((yv_all, np.array([yv_temp])))
        zv_all = np.concatenate((zv_all, np.array([zv_temp])))

xyz_all = np.stack((xv_all, yv_all, zv_all), axis = 1) # Nv x 3 array for all vertex coordinates

Nv = Nv_temp

A0 = np.zeros((Nf, ), dtype = 'float64') # area for all triangles, follow the same order of f_unit

for nf in np.arange(Nf):
    v1 = f_unit[nf, 0]
    v2 = f_unit[nf, 1]
    v3 = f_unit[nf, 2]
    r1 = np.array([xv_all[v1], yv_all[v1], zv_all[v1]])
    r2 = np.array([xv_all[v2], yv_all[v2], zv_all[v2]])
    r3 = np.array([xv_all[v3], yv_all[v3], zv_all[v3]])
    r12 = r2 - r1 # vector from site 1 to 2
    r13 = r3 - r1 # vector from site 1 to 3
    Avec = np.cross(r12, r13) / 2.
    A0[nf] = np.sqrt(np.sum(np.square(Avec)))

# each prism has 6 trapezoids/rectangles, each trapezoid/rectangle has 4 triangles
# in total, each prism has 14 vertices
# vv_list has the list of vertex indices for all prisms, 14 x number of prisms array
Nv0 = 2 * N_cell * (N_row + 1)
Nv1 = Nv0 + N_cell * (N_row + 1)
Nv2 = Nv1 + N_cell * N_row
Nv3 = Nv2 + N_cell * N_row
vv_list = np.zeros((14, N_cell * N_row), dtype = 'int16')
v_count = 0
for nr in np.arange(N_row):
    v_offset = nr * N_cell
    for nc in np.arange(N_cell):
        nc_ift = ift_temp[nc]
        vv_list[:, v_count] = np.array([nc, nc_ift, nc_ift + N_cell, nc + N_cell,
                                        nc + 2 * N_cell, nc_ift + 2 * N_cell, nc_ift + 3 * N_cell, nc + 3 * N_cell,
                                        Nv0 + nc, Nv0 + nc + N_cell,
                                        Nv1 + nc, Nv2 + nc,
                                        Nv3 + nc, Nv3 + nc_ift]) + v_offset * np.concatenate((2 * np.ones((8, )), np.ones((6, ))))
        v_count = v_count + 1

# lists of triangle vertex indices for each prism
# ordered in a way so that each triangle area normal is pointing outwards
# has to be consistent with the way of constructing vv_list
f_v_unit = np.array([[1, 2, 9], [2, 3, 9], [3, 4, 9], [4, 1, 9],
                    [6, 5, 10], [7, 6, 10], [8, 7, 10], [5, 8, 10],
                    [5, 6, 11], [6, 2, 11], [2, 1, 11], [1, 5, 11],
                    [7, 8, 12], [3, 7, 12], [4, 3, 12], [8, 4, 12],
                    [8, 5, 13], [5, 1, 13], [1, 4, 13], [4, 8, 13],
                    [6, 7, 14], [2, 6, 14], [3, 2, 14], [7, 3, 14]], dtype = 'int16') - 1

Nvol = v_count
V0 = np.zeros((Nvol, ), dtype = 'float64') # volume for all prisms, follow the same order of vv_list
for vit in np.arange(Nvol):
    vv_list_v = vv_list[:, vit]
    xyz_vol = np.stack((xv_all[vv_list_v], yv_all[vv_list_v], zv_all[vv_list_v]), axis = 1)
    r1 = xyz_vol[f_v_unit[:, 0], :]
    r2 = xyz_vol[f_v_unit[:, 1], :]
    r3 = xyz_vol[f_v_unit[:, 2], :]

    # Area Force
    r12 = r2 - r1 # vector from site 1 to 2
    r13 = r3 - r1 # vector from site 1 to 3
    Avec = np.cross(r12, r13) / 2.
    V0[vit] = np.sum(Avec * r1) / 3.

# %%
ConfigPlot(xv_all, yv_all, zv_all, f_unit)