"""
Script to visualize flows stored in JHTD data
"""


import h5py
import numpy as np
import os
import library.display.graph as graph
import library.image_processing.movie as movie
import matplotlib.pyplot as plt

# file architecture
savedir = '/Volumes/bigraid/takumi/turbulence/JHTD/analysisresults/'

#Data file
filename = '/Volumes/bigraid/takumi/turbulence/JHTD/isotropic1024coarse_t0_0_tl_2507_x0_0_xl_1024_y0_0_yl_1024_z0_462_zl_101_tstep_20_xstep_1_ystep_1_zstep_1.h5'
filename_no_ext, ext = os.path.splitext(filename)
filename_short = os.path.split(filename_no_ext)
datafilename = filename_short[1]
resultdir = savedir + datafilename + '/'

# plotting style
cmap = 'RdBu'

# JHTD
tau_eta = 0.0446 # Kolmogorov time scale (unit time)
dt = 0.0002 # unit time per a DNS step
fx = 2 * np.pi / 1024 # unit length / px
vmax = np.pi / fx  # px / unit time


fyle = h5py.File(filename, mode='r')
keys = fyle.keys()
# Get keys related to velocities
vel_keys = [s for s in keys if s.startswith('u')]
# Sort velocity-related keys
vel_keys.sort()
tdim = len(vel_keys)
zdim, ydim, xdim, ncomps = fyle[vel_keys[0]].shape
# z-position of illuminated plane
zpos = (zdim + 1) / 2


x, y = np.arange(xdim), np.arange(ydim)
xgrid, ygrid = np.meshgrid(x, y)

for i in range(tdim):
    print '%d / %d' % (i+1, tdim)
    uz, uy, ux = fyle[vel_keys[i]][zpos, ..., 0], fyle[vel_keys[i]][zpos, ..., 1], fyle[vel_keys[i]][zpos, ..., 2] # unit length / unit time
    uz = uz / fx # px/unit time
    uy = uy / fx # px/unit time
    ux = ux / fx # px/unit time

    fig1, ax11, cc11 = graph.color_plot(x, y, ux, vmin=-vmax, vmax=vmax, cmap=cmap, subplot=131, figsize=(20, 6))
    fig1, ax12, cc12 = graph.color_plot(x, y, uy, vmin=-vmax, vmax=vmax, cmap=cmap, subplot=132, figsize=(20, 6))
    fig1, ax13, cc13 = graph.color_plot(x, y, uz, vmin=-vmax, vmax=vmax, cmap=cmap, subplot=133, figsize=(20, 6))
    axes = [ax11, ax12, ax13]
    ccs = [cc11, cc12, cc13]
    clabels = ['$u_x$ (px/unit time)', '$u_y$ (px/unit time)', '$u_z$ (px/unit time)']
    for ax, cc, clabel in zip(axes, ccs, clabels):
        graph.add_colorbar(cc, ax=ax, label=clabel)
        graph.labelaxes(ax, '$X$ (px)', '$Y$ (px)')
    graph.suptitle(r't= %.1f $ \tau_\eta$ = %.2f (unit time)' % (float(vel_keys[i][1:]) * dt / tau_eta, float(vel_keys[i][1:]) * dt), fignum=1)
    #
    fig2, ax2 = graph.pdf(ux, nbins=int(np.sqrt(ydim * xdim)), fignum=2, label='$u_x$ (px/unit time)')
    fig2, ax2 = graph.pdf(uy, nbins=int(np.sqrt(ydim * xdim)), fignum=2, label='$u_y$ (px/unit time)')
    fig2, ax2 = graph.pdf(uz, nbins=int(np.sqrt(ydim * xdim)), fignum=2, label='$u_z$ (px/unit time)')

    graph.setaxes(ax2, -vmax, vmax, -0.05/100., 0.006)
    graph.labelaxes(ax2, '$u_i$ (px/unit time)', 'Probability density')
    graph.suptitle(r't= %.1f $ \tau_\eta$ = %.2f (unit time)' % (float(vel_keys[i][1:]) * dt / tau_eta, float(vel_keys[i][1:]) * dt), fignum=2)
    ax2.legend()

    filename1 = 'vel_field_t' + vel_keys[i][1:]
    filename2 = 'pdf_vel_field_t' + vel_keys[i][1:]
    graph.save(resultdir + filename1, fignum=1, ext='png')
    graph.save(resultdir + filename2, fignum=2, ext='png')

    plt.close('all')
fyle.close()

# # make movies
# movie.make_movie(resultdir + 'vel_field_t', resultdir + 'movie_vel_field', ext='png', framerate=10, option='glob')
# movie.make_movie(resultdir + 'pdf_vel_field_t', resultdir + 'movie_pdf_vel_field', ext='png', framerate=10, option='glob')