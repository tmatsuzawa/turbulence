"""
Compare pivlab outputs with synthetic data for unidirectional velocity fields
... Generates heatmap of PIV outputs, original velocity field, difference for Ux and Uy
"""

import argparse
import glob
import os
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import library.tools.rw_data as rw
import turbulence.jhtd.get as jhtd_get
import turbulence.jhtd.tools as jhtd_tools
import library.basics.formatarray as fa
import library.basics.formatstring as fs
import library.display.graph as graph
import library.image_processing.movie as movie

parser = argparse.ArgumentParser(description='Make a hdf5 file out of PIVLab txt outputs')
parser.add_argument('-datadir', '--datadir', help='Parent directory of PIVlab output HDF5 directories', type=str,
                    default='/Volumes/labshared3-1/takumi/JHTD-sample/JHT_Database/Data/synthetic_data_unidirectional/hdf5data/')
parser.add_argument('-fdatadir', '--fakedatadir', help='full path to fake data', type=str,
                    default='/Volumes/labshared3-1/takumi/JHTD-sample/JHT_Database/Data/synthetic_data_unidirectional/org_vel_field/')
parser.add_argument('-mode', '--mode', help='Choose from constant and gradient', type=str,
                    default='constant')
args = parser.parse_args()

# Data architecture
parentdirs = glob.glob(args.datadir + '*')
fakedata = glob.glob(args.fakedatadir + 'unidirectional*')
resultdirname = 'comparison_pivlab_org' # name of directory where figures will be stored

# Plotting settings Part 1
cmap = 'RdBu'
cmap2 = 'Blues'
params = {'figure.figsize': (24, 10)
          }
graph.update_figure_params(params)


for parentdir in parentdirs:
    print parentdir
    iw = int(fs.get_float_from_str(parentdir, 'W', 'px')) # interrogation width
    pivdata_loc = glob.glob(parentdir + '/PIVlab*')
    resultdir = os.path.join(parentdir, resultdirname)

    for pivdatum_loc in pivdata_loc:
        # Extract desired displacement
        if args.mode == 'constant':
            mag = fs.get_float_from_str(pivdatum_loc, 'mag', '.h5')  # velocity magnitude
            mag_str = fs.convert_float_to_decimalstr(mag)
            mag_str_2 = fs.convert_float_to_decimalstr(mag, zpad=4) # for naming scheme


            # Plotting settings Part 2
            vmin, vmax = mag * 0, mag * 1.2

            # Load pivlab output
            pivdata = rw.read_hdf5(pivdatum_loc)
            xx, yy = pivdata['x'], pivdata['y']
            ux, uy = pivdata['ux'][..., 0], pivdata['uy'][..., 0]

            fig1, ax11, cc11 = graph.color_plot(xx, yy, ux, vmin=vmin, vmax=vmax, cmap=cmap2, fignum=1, subplot=231)
            fig1, ax14, cc14 = graph.color_plot(xx, yy, uy, vmin=-1, vmax=1, cmap=cmap, fignum=1, subplot=234)

            # Find an original velocity field data
            for fakedatum in fakedata:
                if mag_str in fakedatum:
                    print 'Fake data found!'
                    # Load fake data
                    fdata = rw.read_hdf5(fakedatum)
                    xx0, yy0 = fdata['x'], fdata['y']
                    ux0, uy0 = fdata['ux'], fdata['uy']
                    fig1, ax12, cc12 = graph.color_plot(xx0, yy0, ux0, vmin=vmin, vmax=vmax, cmap=cmap2, fignum=1, subplot=232)
                    fig1, ax15, cc15 = graph.color_plot(xx0, yy0, uy0, vmin=-1, vmax=1, cmap=cmap, fignum=1, subplot=235)
                    # coarse_grain data
                    xx_c = fa.coarse_grain_2darr_overwrap(xx0, iw, iw, overwrap=0.5)
                    yy_c = fa.coarse_grain_2darr_overwrap(yy0, iw, iw, overwrap=0.5)
                    ux_c = fa.coarse_grain_2darr_overwrap(ux0, iw, iw, overwrap=0.5)
                    uy_c = fa.coarse_grain_2darr_overwrap(uy0, iw, iw, overwrap=0.5)
                    fig1, ax13, cc13 = graph.color_plot(xx, yy, ux - ux_c, vmin=-1,vmax=1, cmap=cmap, fignum=1, subplot=233)
                    fig1, ax16, cc16 = graph.color_plot(xx, yy, uy - uy_c, vmin=-1, vmax=1, cmap=cmap, fignum=1, subplot=236)

                    # Plotting stuff
                    axes = [ax11, ax12, ax13, ax14, ax15, ax16]
                    ccs = [cc11, cc12, cc13, cc14, cc15, cc16]
                    cblabels = [r'$U_x$', r'$U_x$', r'$U_x$', r'$U_y$', r'$U_y$', r'$U_y$']
                    xlabel, ylabel = r'$X$ (px)', r'$Y$ (px)'
                    title = r'W=%dpx, $v$=%.1f px/frame (PIVLab, Original, Diff.)' % (iw, mag)
                    for ax, cc, cblabel in zip(axes, ccs, cblabels):
                        graph.add_colorbar(cc, ax=ax, label=cblabel)
                        graph.labelaxes(ax, xlabel, ylabel)
                        graph.setaxes(ax, 0, xx0.shape[1], 0, xx0.shape[0])
                    graph.suptitle(title, fignum=1)
                    fig1.tight_layout()

                    pivdata.close()
                    fdata.close()

                    figname = '/pivlab_fakedata_comp_W%d_v%spxframe_multiple_passes' % (iw, mag_str_2)
                    graph.save(resultdir + figname, ext='png')
                    plt.close()
                    break
        elif args.mode == 'gradient':
            max = fs.get_float_from_str(pivdatum_loc, 'max', '_min')  # max velocity
            min = fs.get_float_from_str(pivdatum_loc, 'min', '.h5')  # min velocity

            max_str = 'max' + fs.convert_float_to_decimalstr(max)
            min_str = 'min' + fs.convert_float_to_decimalstr(min)
            max_str_2 = 'max' + fs.convert_float_to_decimalstr(max, zpad=4) # for naming scheme
            min_str_2 = 'min' + fs.convert_float_to_decimalstr(min, zpad=4) # for naming scheme

            # Plotting settings Part 2
            vmin, vmax = min, max * 1.2

            # Load pivlab output
            pivdata = rw.read_hdf5(pivdatum_loc)
            xx, yy = pivdata['x'], pivdata['y']
            ux, uy = pivdata['ux'][..., 0], pivdata['uy'][..., 0]

            fig1, ax11, cc11 = graph.color_plot(xx, yy, ux, vmin=vmin, vmax=vmax, cmap=cmap2, fignum=1, subplot=231)
            fig1, ax14, cc14 = graph.color_plot(xx, yy, uy, vmin=-1, vmax=1, cmap=cmap, fignum=1, subplot=234)

            # Find an original velocity field data
            for fakedatum in fakedata:
                if max_str in fakedatum and min_str in fakedatum:
                    print 'Fake data found!'
                    # Load fake data
                    fdata = rw.read_hdf5(fakedatum)
                    xx0, yy0 = fdata['x'], fdata['y']
                    ux0, uy0 = fdata['ux'], fdata['uy']
                    fig1, ax12, cc12 = graph.color_plot(xx0, yy0, ux0, vmin=vmin, vmax=vmax, cmap=cmap2, fignum=1,
                                                        subplot=232)
                    fig1, ax15, cc15 = graph.color_plot(xx0, yy0, uy0, vmin=-1, vmax=1, cmap=cmap, fignum=1,
                                                        subplot=235)
                    # coarse_grain data
                    xx_c = fa.coarse_grain_2darr_overwrap(xx0, iw, iw, overwrap=0.5)
                    yy_c = fa.coarse_grain_2darr_overwrap(yy0, iw, iw, overwrap=0.5)
                    ux_c = fa.coarse_grain_2darr_overwrap(ux0, iw, iw, overwrap=0.5)
                    uy_c = fa.coarse_grain_2darr_overwrap(uy0, iw, iw, overwrap=0.5)
                    fig1, ax13, cc13 = graph.color_plot(xx, yy, ux - ux_c, vmin=-1, vmax=1, cmap=cmap, fignum=1,
                                                        subplot=233)
                    fig1, ax16, cc16 = graph.color_plot(xx, yy, uy - uy_c, vmin=-1, vmax=1, cmap=cmap, fignum=1,
                                                        subplot=236)

                    # Plotting stuff
                    axes = [ax11, ax12, ax13, ax14, ax15, ax16]
                    ccs = [cc11, cc12, cc13, cc14, cc15, cc16]
                    cblabels = [r'$U_x$', r'$U_x$', r'$\Delta U_x$', r'$U_y$', r'$U_y$', r'$\Delta U_y$']
                    xlabel, ylabel = r'$X$ (px)', r'$Y$ (px)'
                    title = r'W=%dpx, $v_{max}$=%.1f px/frame (PIVLab, Original, Diff.)' % (iw, max)
                    for ax, cc, cblabel in zip(axes, ccs, cblabels):
                        graph.add_colorbar(cc, ax=ax, label=cblabel)
                        graph.labelaxes(ax, xlabel, ylabel)
                        graph.setaxes(ax, 0, xx0.shape[1], 0, xx0.shape[0])
                    graph.suptitle(title, fignum=1)
                    fig1.tight_layout()

                    pivdata.close()
                    fdata.close()

                    figname = '/pivlab_fakedata_comp_W%d_vmax%spxframe_vmin%spxframe_multiple_passes' % (iw, max_str_2, min_str_2)
                    graph.save(resultdir + figname, ext='png')
                    plt.close()
                    break

    imgfiles = resultdir + figname[:6]
    moviename = '/movie_W%dpx_multiple_passes' % iw
    movie.make_movie(imgfiles, resultdir + moviename, framerate=1, option='glob', ext='png')

print '... Done!'