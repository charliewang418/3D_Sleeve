#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm

def ConfigPlot(x, y, z, f_unit, mark_print = 0, fig_name = ''):
    Nf = f_unit.shape[0]
    cmap = plt.get_cmap('turbo', Nf)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    cidx = np.random.permutation(Nf)

    # plot each triangle with a random color
    for nf in np.arange(Nf): 
        f_sub = f_unit[nf, :]
        xv = x[f_sub]
        yv = y[f_sub]
        zv = z[f_sub]
        verts = [list(zip(xv, yv, zv))]
        srf = Poly3DCollection(verts, alpha = 0.8, facecolor = cmap.colors[cidx[nf]])
        ax.add_collection3d(srf)

    # plot vertex connections in black lines
    for f_sub in f_unit:
        xv = x[f_sub]
        yv = y[f_sub]
        zv = z[f_sub]
        ax.plot([xv[0], xv[1]], [yv[0], yv[1]], [zv[0], zv[1]], c = 'black', linewidth = 2)
        ax.plot([xv[1], xv[2]], [yv[1], yv[2]], [zv[1], zv[2]], c = 'black', linewidth = 2)
        ax.plot([xv[2], xv[0]], [yv[2], yv[0]], [zv[2], zv[0]], c = 'black', linewidth = 2)

    # plot vertices in blue dots
    for xv, yv, zv in zip(x, y, z):
        ax.scatter(xv, yv, zv, c = 'blue', s = 30)

    # whether or not to save the figure
    if (mark_print == 1) and (not fig_name):
        fig.savefig(fig_name, dpi = 300)

    plt.show()