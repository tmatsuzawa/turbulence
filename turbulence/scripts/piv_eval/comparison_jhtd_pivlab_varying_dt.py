"""

"""

import argparse
import glob
import os
import numpy as np
import h5py
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import library.tools.rw_data as rw
import turbulence.jhtd.get as jhtd_get
import turbulence.jhtd.tools as jhtd_tools
import library.basics.formatarray as fa
import library.basics.formatstring as fs
import library.display.graph as graph
import sys
import time

parser = argparse.ArgumentParser(description='Make a hdf5 file out of PIVLab txt outputs')
parser.add_argument('-datadir', '--datadir', help='Parent directory of PIVlab output HDF5 directories', type=str,
                    default='/Volumes/bigraid/takumi/turbulence/JHTD/synthetic_data/hdf5data/tstep1_npt50000_lt20p0_pbcTrue_no_uz_varyingDt/post_processed/')
parser.add_argument('-jhtd', '--jhtddatapath', help='full path to jhtd data', type=str,
                    default='/Volumes/bigraid/takumi/turbulence/JHTD/isotropic1024coarse_t0_0_tl_100_x0_0_xl_1024_y0_0_yl_1024_z0_462_zl_101_tstep_1_xstep_1_ystep_1_zstep_1.h5')
parser.add_argument('-overwrite', help='Redo computation, and overwrite figures and data', action='store_true')

args = parser.parse_args()

def get_jhtd_parameters(filename):
    keys = ['t0', 'tl', 'x0', 'xl', 'y0', 'yl', 'z0', 'zl', 'tstep', 'xstep', 'ystep', 'zstep']
    print filename
    param = {}
    for key in keys:
        try:
            param[key] = int(fs.get_float_from_str(filename, key + '_', '_'))
        except RuntimeError:
            print '... RuntimeError raised. Try something else...'
            param[key] = int(fs.get_float_from_str(filename, key + '_', '.h5'))
            print '... Worked. '
    return param


# Data architecture
jhtddatapath = args.jhtddatapath
pivdatadir = args.datadir
parentdir = os.path.dirname(pivdatadir)


# Plotting settings
cmap = 'RdBu'
# cmap = 'magma'
params = {'figure.figsize': (20, 20)
          }
graph.update_figure_params(params)
err_ratio_max, err_ratio_min = 100, -100

# jhtd parameters
param = get_jhtd_parameters(jhtddatapath)
# dx = 2 * np.pi / 1024. # unit length / px
dy = dz = dx = 1 # in px
dt_sim = 0.0002  # DNS simulation time step
if 'coarse' in jhtddatapath:
    dt_spacing = 10
else:
    dt_spacing = 1
# dt = dt_sim * param['tstep'] * dt_spacing # time separation between data points in JHTD time unit
nu = 0.000185  # viscosity (unit length ^2 / unit time)
fx = (2 * np.pi) / 1024 # unit length / px

# print 'dt, param[tstep], dt_spacing:', dt, param['tstep'], dt_spacing

# Make convienient lists about time and space
T = range(param['t0'] * dt_spacing, param['tl'] * dt_spacing, param['tstep'] * dt_spacing)
X = range(param['x0'], param['x0'] + param['xl'], param['xstep'])
Y = range(param['y0'], param['y0'] + param['yl'], param['ystep'])
Z = range(param['z0'], param['z0'] + param['zl'], param['zstep'])
zpos = (param['zl']+1)/2 # index specifying where center of the illumination plane
tdim, zdim, ydim, xdim = len(T), len(Z), len(Y), len(X)
# mesh grid for pcolormesh
xx, yy = np.meshgrid(X, Y)


# PIVLab outputs
pivlab_outputs = glob.glob(pivdatadir + '*.h5') # generated by pivlab2hdf5.py




# Processing starts now...


# Load jhtd data
# jhtd_data[vel_key] has a structure (z, y, x, ui)... vel_key is time
jhtd_data = h5py.File(args.jhtddatapath, mode='r')
vel_keys = [jhtd_key for jhtd_key in jhtd_data.keys() if jhtd_key.startswith('u')]

field_shape = np.shape(jhtd_data[vel_keys[0]])

# print 'key, ', vel_keys[0]
# print 'shape, ', np.shape(jhtd_data[vel_keys[0]][field_shape[0]/2, :, :, 2])
# print 'type, ', type(jhtd_data[vel_keys[0]])
# import sys
# sys.exit()

for i, pivlab_output in enumerate(pivlab_outputs):
    print pivlab_output
    deltat = int(fs.get_float_from_str(pivlab_output, 'Dt_', '_npt')) #number of frames between image A and image B
    dt = dt_sim * deltat * dt_spacing  # time separation between data points in JHTD time unit

    # Plotting settings 2
    # Conversion between JHTD unit system and PIVLab unit system
    # vmax = np.pi (length unit / time unit) = 512 (px/time unit) = 512 px/time unit * (1 time unit / 5000 DNS steps) * (param['tstep'] * dt_spacing DNS step  / 1 frame)
    # 1 frame = param['tstep'] * dt_spacing  DNS steps
    vmax = max(param['xl'], param['yl']) / 2 * dt_sim * (deltat * dt_spacing)  # px/frame
    vmin = - vmax

    # File architecture 2
    resultsubdirname = fs.get_filename_wo_ext(pivlab_output)
    resultdir = os.path.join(parentdir, 'comp_jhtd_pivlab_local/' + resultsubdirname)


    iw = int(fs.get_float_from_str(pivlab_output, 'W', 'pix'))  # size of final interrogation window in multi-pass
    lt = fs.get_float_from_str(pivlab_output, 'lt', '_')  # laser thickness in px
    npt = int(fs.get_float_from_str(pivlab_output, 'npt', '_'))  # number of particles used to generate fake piv images
    fwhm = lt / np.sqrt(2) # fwhm of a gaussian beam

    # field averaged over illuminated plane thickness
    print 'Center of illuminated plane %d, FWHM of a Gaussian beam: %.2f' % (zpos, fwhm)
    z_start, z_end = max(int(zpos - fwhm/2), 0), min(int(zpos + fwhm/2), zdim-1)
    print 'z of illuminatied volume: [%d, %d] (px)' % (z_start, z_end)


    # graph.color_plot(xx, yy, ux_mean, fignum=1)
    # graph.color_plot(xx, yy, ux_center, fignum=2)

    # Load pivlab-processed data
    # piv_data has a structure (x, y, t)
    piv_data = h5py.File(pivlab_output, mode = 'r')


    # Coarse-grain
    nrows_sub, ncolumns_sub = iw, iw  # number of pixels to average over
    xx_coarse = fa.coarse_grain_2darr_overwrap(xx, nrows_sub, ncolumns_sub, overwrap=0.5)
    yy_coarse = fa.coarse_grain_2darr_overwrap(yy, nrows_sub, ncolumns_sub, overwrap=0.5)

    for j, t in enumerate(range(0, tdim-deltat, deltat)):
        print '%d-%d / %d' % (t, t+deltat, tdim-1)
        print range(0, tdim-deltat, deltat)

        # If img already exists, skip
        imgfilename1 = 'comp_pivlab_and_jhtd_field_at_center_of_illum_plane/' \
                       'comp_pivlab_and_jhtd_field_at_center_of_illum_planeim_{0:04d}'.format(t)
        imgfilename2 = 'comp_pivlab_and_jhtd_field_avg_over_illum_volume_in_z_direction/' \
                       'comp_pivlab_and_jhtd_field_avg_over_illum_volume_in_z_directionim_{0:04d}'.format(t)
        # if os.path.exists(os.path.join(parentdir, 'pdf_data/' + resultsubdirname + '.h5')) and not args.overwrite:
        #     print '... data already exists! Skipping...'
        #     continue


        # Initialize
        uy_mean_temp = np.zeros_like(xx_coarse)
        ux_mean_temp = np.zeros_like(xx_coarse)
        uy_center_temp = np.zeros_like(xx_coarse)
        ux_center_temp = np.zeros_like(xx_coarse)


        # average and coarse-grain jhtd fields
        for tt in range(t, t+deltat, 1):
            print 'Averaging jhtd vel fields... %d / %d' % (tt+1, t+deltat)
            print '... averaging jhtd vel fields '
            t1 = time.time()
            # uz_mean = np.nanmean(jhtd_data[vel_keys[tt]][z_start:z_end, ..., 0], axis=0)
            uy_mean = np.nanmean(jhtd_data[vel_keys[tt]][z_start:z_end, ..., 1], axis=0) / fx * dt # length unit/time unit -> px/frame
            ux_mean = np.nanmean(jhtd_data[vel_keys[tt]][z_start:z_end, ..., 2], axis=0) / fx * dt # length unit/time unit -> px/frame
            t2 = time.time()
            print '... time took to average 2 fields: %.3f s' % (t2-t1)


            # jhtd field at the center of illuminated plane
            # uz_center = jhtd_data[vel_keys[tt]][zpos, ..., 0]
            uy_center = jhtd_data[vel_keys[tt]][zpos, ..., 1] / fx * dt # length unit/time unit -> px/frame
            ux_center = jhtd_data[vel_keys[tt]][zpos, ..., 2] / fx * dt # length unit/time unit -> px/frame

            # Coarse-grain
            print '... coarse-graining jhtd vel fields '
            t3 = time.time()
            ux_mean_coarse = fa.coarse_grain_2darr_overwrap(ux_mean, nrows_sub, ncolumns_sub, overwrap=0.5)
            uy_mean_coarse = fa.coarse_grain_2darr_overwrap(uy_mean, nrows_sub, ncolumns_sub, overwrap=0.5)
            ux_center_coarse = fa.coarse_grain_2darr_overwrap(ux_center, nrows_sub, ncolumns_sub, overwrap=0.5)
            uy_center_coarse = fa.coarse_grain_2darr_overwrap(uy_center, nrows_sub, ncolumns_sub, overwrap=0.5)
            t4 = time.time()
            print '... time took to coarse-grain 4 fields: %.3f s' % (t4-t3)
            # For field averaged along z and time
            uy_mean_temp += uy_mean_coarse
            ux_mean_temp += ux_mean_coarse
            uy_center_temp += uy_center_coarse
            ux_center_temp += ux_center_coarse

        # Average over time
        uy_mean_temp /= float(deltat)
        ux_mean_temp /= float(deltat)
        uy_center_temp /= float(deltat)
        ux_center_temp /= float(deltat)


        # pivlab outputs
        ux_pivlab, uy_pivlab = np.array(piv_data['ux']), np.array(piv_data['uy']) # px / frame
        xx_pivlab, yy_pivlab = np.array(piv_data['x']), np.array(piv_data['y'])
        #
        # # PLOTTING
        # # Fig 1: Comparison between PIVLab outputs and a field from JHTD at the CENTER of the illuminated plane
        # # PIVLab Outputs
        # fig1, ax11, cc11 = graph.color_plot(xx_pivlab, yy_pivlab, ux_pivlab[..., j], cmap=cmap, vmin=vmin, vmax=vmax, fignum=1, subplot=331)
        # fig1, ax14, cc14 = graph.color_plot(xx_pivlab, yy_pivlab, uy_pivlab[..., j], cmap=cmap, vmin=vmin, vmax=vmax, fignum=1, subplot=334)
        # # Fields from JHTD
        # fig1, ax12, cc12 = graph.color_plot(xx_coarse, yy_coarse, ux_mean_temp, cmap=cmap, vmin=vmin, vmax=vmax, fignum=1, subplot=332)
        # fig1, ax15, cc15 = graph.color_plot(xx_coarse, yy_coarse, uy_mean_temp, cmap=cmap, vmin=vmin, vmax=vmax, fignum=1, subplot=335)
        # # Difference (error)
        # fig1, ax13, cc13 = graph.color_plot(xx_coarse, yy_coarse, (ux_pivlab[..., j] - ux_mean_temp) / ux_mean_temp, cmap=cmap, vmin=vmin, vmax=vmax, fignum=1, subplot=333)
        # fig1, ax16, cc16 = graph.color_plot(xx_coarse, yy_coarse, (uy_pivlab[..., j] - uy_mean_temp) / uy_mean_temp, cmap=cmap, vmin=vmin, vmax=vmax, fignum=1, subplot=336)
        # # PDF
        # nbins = int(np.sqrt(ux_center_coarse.shape[0] * ux_center_coarse.shape[1]))
        # nbins_err = int((err_ratio_max - err_ratio_min) * 10)
        # fig1, ax17, bins17ux, hist17ux = graph.pdf(ux_pivlab[..., j], nbins=nbins, fignum=1, subplot=337, label=r'$U_x$', return_data=True)
        # fig1, ax17, bins17uy, hist17uy = graph.pdf(uy_pivlab[..., j], nbins=nbins, fignum=1, subplot=337, label=r'$U_y$', return_data=True)
        # fig1, ax18, bins18ux, hist18ux = graph.pdf(ux_center_coarse, nbins=nbins,  fignum=1, subplot=338, label=r'$U_x$', return_data=True)
        # fig1, ax18, bins18uy, hist18uy = graph.pdf(uy_center_coarse, nbins=nbins,  fignum=1, subplot=338, label=r'$U_y$', return_data=True)
        # fig1, ax19, bins19ux, hist19ux = graph.pdf((ux_pivlab[..., j] - ux_mean_temp) / ux_mean_temp, nbins=nbins_err,  fignum=1, subplot=339, label=r'$\Delta U_x(\vec{x}) / U_x(\vec{x})$', return_data=True, vmax=err_ratio_max, vmin=err_ratio_min)
        # fig1, ax19, bins19uy, hist19uy = graph.pdf((uy_pivlab[..., j] - uy_mean_temp) / uy_mean_temp, nbins=nbins_err,  fignum=1, subplot=339, label=r'$\Delta U_y(\vec{x}) / U_y(\vec{x})$', return_data=True, vmax=err_ratio_max, vmin=err_ratio_min)
        #
        # # Fig 2: Comparison between PIVLab outputs and a field from JHTD on the illuminated plane, AVERAGED over laser thickness
        # fig2, ax21, cc21 = graph.color_plot(xx_pivlab, yy_pivlab, ux_pivlab[..., j], cmap=cmap, vmin=vmin, vmax=vmax, fignum=2, subplot=331)
        # fig2, ax24, cc24 = graph.color_plot(xx_pivlab, yy_pivlab, uy_pivlab[..., j], cmap=cmap, vmin=vmin, vmax=vmax, fignum=2, subplot=334)
        # # Fields from JHTD
        # fig2, ax22, cc22 = graph.color_plot(xx_coarse, yy_coarse, ux_center_temp, cmap=cmap, vmin=vmin, vmax=vmax, fignum=2, subplot=332)
        # fig2, ax25, cc25 = graph.color_plot(xx_coarse, yy_coarse, uy_center_temp, cmap=cmap, vmin=vmin, vmax=vmax, fignum=2, subplot=335)
        # # Difference (error)
        # fig2, ax23, cc23 = graph.color_plot(xx_coarse, yy_coarse, (ux_pivlab[..., j] - ux_center_temp) / ux_center_temp, cmap=cmap, vmin=vmin, vmax=vmax, fignum=2, subplot=333)
        # fig2, ax26, cc26 = graph.color_plot(xx_coarse, yy_coarse, (uy_pivlab[..., j] - uy_center_temp) / uy_center_temp, cmap=cmap, vmin=vmin, vmax=vmax, fignum=2, subplot=336)
        # # PDF
        # fig2, ax27 = graph.pdf(ux_pivlab[..., j], nbins=nbins,  fignum=2, subplot=337, label=r'$U_x$')
        # fig2, ax27 = graph.pdf(uy_pivlab[..., j], nbins=nbins,  fignum=2, subplot=337, label=r'$U_y$')
        # fig2, ax28, bins28ux, hist28ux = graph.pdf(ux_mean_coarse, nbins=nbins,  fignum=2, subplot=338, label=r'$U_x$', return_data=True)
        # fig2, ax28, bins28uy, hist28uy = graph.pdf(uy_mean_coarse, nbins=nbins,  fignum=2, subplot=338, label=r'$U_y$', return_data=True)
        # fig2, ax29, bins29ux, hist29ux = graph.pdf((ux_pivlab[..., j] - ux_center_temp) / ux_center_temp, nbins=nbins_err,  fignum=2, subplot=339, label=r'$\Delta U_x(\vec{x}) / U_x(\vec{x})$', return_data=True, vmax=err_ratio_max, vmin=err_ratio_min)
        # fig2, ax29, bins29uy, hist29uy = graph.pdf((uy_pivlab[..., j] - uy_center_temp) / uy_center_temp, nbins=nbins_err,  fignum=2, subplot=339, label=r'$\Delta U_y(\vec{x}) / U_y(\vec{x})$', return_data=True, vmax=err_ratio_max, vmin=err_ratio_min)
        #
        # cc1s = [cc11, cc12, cc13, cc14, cc15, cc16, None, None, None]
        # cc2s = [cc21, cc22, cc23, cc24, cc25, cc26, None, None, None]
        # ax1s = [ax11, ax12, ax13, ax14, ax15, ax16, ax17, ax18, ax19]
        # ax2s = [ax21, ax22, ax23, ax24, ax25, ax26, ax27, ax28, ax29]
        # add_legends = [False, False, False, False, False, False, True, True, True]
        # add_cbs = [True, True, True, True, True, True, False, False, False]
        # xlabels = [r'$X$ (px)'] * 6 + [r'$U_i$ (px/frame)', r'$U_i$ (px/frame)', r'$\Delta U_i(\vec{x}) / U_i(\vec{x})$']
        # ylabels = [r'$Y$ (px)'] * 6 + ['Probabilistic density'] * 3
        # clabels = [r'$U_x$ (px/frame)', r'$U_x$ (px/frame)', r'$\Delta U_x(\vec{x}) / U_x(\vec{x})$',
        #            r'$U_y$ (px/frame)', r'$U_y$ (px/frame)', r'$\Delta U_y(\vec{x}) / U_y(\vec{x})$', None, None, None]


        ux_err_mean_temp = ux_pivlab[..., j] - ux_mean_temp
        uy_err_mean_temp = uy_pivlab[..., j] - uy_mean_temp
        ux_err_center_temp = ux_pivlab[..., j] - ux_center_temp
        uy_err_center_temp = uy_pivlab[..., j] - uy_center_temp

        # # Prepare Fig 1
        # for ax, cc, xlabel, ylabel, clabel, add_legend, add_cb in zip(ax1s, cc1s, xlabels, ylabels, clabels, add_legends, add_cbs):
        #     graph.labelaxes(ax, xlabel, ylabel)
        #     if add_cb:
        #         graph.add_colorbar(cc, ax=ax, label=clabel)
        #     if add_legend:
        #         ax.legend()
        #     if ax in [ax17, ax18]:
        #         # graph.setaxes(ax, -16, 16, -0.005, 0.18)
        #         ax.set_xlim(-vmax, vmax)
        #     elif ax is ax19:
        #         ax.set_xlim(err_ratio_min, err_ratio_max)
        #     else:
        #         ax.set_facecolor('k')
        #
        # # Prepare Fig 2
        # for ax, cc, xlabel, ylabel, clabel, add_legend, add_cb in zip(ax2s, cc2s, xlabels, ylabels, clabels, add_legends, add_cbs):
        #     graph.labelaxes(ax, xlabel, ylabel)
        #     if add_cb:
        #         graph.add_colorbar(cc, ax=ax, label=clabel)
        #     if add_legend:
        #         ax.legend()
        #     if ax in [ax27, ax28]:
        #         # graph.setaxes(ax, -16, 16, -0.005, 0.18)
        #         ax.set_xlim(-vmax, vmax)
        #     elif ax is ax29:
        #         ax.set_xlim(err_ratio_min, err_ratio_max)
        #     else:
        #         ax.set_facecolor('k')
        # title = r'$W = {0:d}$ px, FWHM={1:.1f} px, $D=$3px, $N={2:d}$, $Re_\lambda$=420, $t={3:05.3f}$, 1 frame={4:.3f}'.format(iw, fwhm, npt, t * dt_sim * param['tstep'] * dt_spacing, dt)
        # graph.suptitle(title, fignum=1)
        # graph.suptitle(title, fignum=2)
        # fig1.tight_layout()
        # fig2.tight_layout()
        #
        #
        # # Save images
        # graph.save(os.path.join(resultdir, imgfilename1), fignum=1, ext='png')
        # graph.save(os.path.join(resultdir, imgfilename2), fignum=2, ext='png')
        # plt.close('all')

        # Data management
        if j == 0:
            # bins_ux_pivlab, hist_ux_pivlab, bins_uy_pivlab, hist_uy_pivlab = bins17ux, hist17ux, bins17uy, hist17uy
            # bins_ux_jhtd_c, hist_ux_jhtd_c, bins_uy_jhtd_c, hist_uy_jhtd_c = bins18ux, hist18ux, bins18uy, hist18uy
            # bins_ux_diff_c, hist_ux_diff_c, bins_uy_diff_c, hist_uy_diff_c = bins19ux, hist19ux, bins19uy, hist19uy
            # bins_ux_jhtd_avg, hist_ux_jhtd_avg, bins_uy_jhtd_avg, hist_uy_jhtd_avg = bins28ux, hist28ux, bins28uy, hist28uy
            # bins_ux_diff_avg, hist_ux_diff_avg, bins_uy_diff_avg, hist_uy_diff_avg = bins29ux, hist29ux, bins29uy, hist29uy
            # histdata_list = [bins_ux_pivlab, hist_ux_pivlab, bins_uy_pivlab, hist_uy_pivlab,
            #              bins_ux_jhtd_c, hist_ux_jhtd_c, bins_uy_jhtd_c, hist_uy_jhtd_c,
            #              bins_ux_diff_c, hist_ux_diff_c, bins_uy_diff_c, hist_uy_diff_c,
            #              bins_ux_jhtd_avg, hist_ux_jhtd_avg, bins_uy_jhtd_avg, hist_uy_jhtd_avg,
            #              bins_ux_diff_avg, hist_ux_diff_avg, bins_uy_diff_avg, hist_uy_diff_avg]
            #
            errdata_list = [ux_mean_temp, uy_mean_temp, ux_center_temp, uy_center_temp,
                            ux_err_mean_temp, uy_err_mean_temp, ux_err_center_temp, uy_err_center_temp]

        else:
            # histdata_list_tmp = [bins17ux, hist17ux, bins17uy, hist17uy,
            #                      bins18ux, hist18ux, bins18uy, hist18uy,
            #                      bins19ux, hist19ux, bins19uy, hist19uy,
            #                      bins28ux, hist28ux, bins28uy, hist28uy,
            #                      bins29ux, hist29ux, bins29uy, hist29uy]
            errdata_list_tmp = [ux_mean_temp, uy_mean_temp, ux_center_temp, uy_center_temp,
                            ux_err_mean_temp, uy_err_mean_temp, ux_err_center_temp, uy_err_center_temp]

            # for k, (content, tmp_data, content_err, tmp_data_err) in enumerate(zip(histdata_list, histdata_list_tmp, errdata_list, errdata_list_tmp)):
            #     # Note that content = np.vstack((content, tmp_data)) does NOT work
            #     # because content and histdata_list probably have different identities.
            #     # substitution to a sequence changes an object but substitution to a variable (content) does NOT.
            #     histdata_list[k] = np.vstack((content, tmp_data))
            #     errdata_list[k] = np.vstack((content_err, tmp_data_err))


            for k, (content_err, tmp_data_err) in enumerate(zip(errdata_list, errdata_list_tmp)):
                # Note that content = np.vstack((content, tmp_data)) does NOT work
                # because content and histdata_list probably have different identities.
                # substitution to a sequence changes an object but substitution to a variable (content) does NOT.
                errdata_list[k] = np.dstack((content_err, tmp_data_err))



        plt.close('all')

    # Save data
    try:
        # histdata = {}
        # datanames = ['bins_ux_pivlab', 'hist_ux_pivlab', 'bins_uy_pivlab', 'hist_uy_pivlab',
        #              'bins_ux_jhtd_c', 'hist_ux_jhtd_c', 'bins_uy_jhtd_c', 'hist_uy_jhtd_c',
        #              'bins_ux_diff_c', 'hist_ux_diff_c', 'bins_uy_diff_c', 'hist_uy_diff_c',
        #              'bins_ux_jhtd_avg', 'hist_ux_jhtd_avg', 'bins_uy_jhtd_avg', 'hist_uy_jhtd_avg',
        #              'bins_ux_diff_avg', 'hist_ux_diff_avg', 'bins_uy_diff_avg', 'hist_uy_diff_avg']
        # for dataname, content in zip(datanames, histdata_list):
        #     histdata[dataname] = content
        #
        # datafilepath = os.path.join(parentdir, 'pdf_data_local/' + resultsubdirname)
        # rw.write_hdf5_dict(datafilepath, histdata)
        #

        err_data = {}
        datanames = ['ux_mean', 'uy_mean', 'ux_center_temp', 'uy_center_temp',
                     'ux_err_mean_temp', 'uy_err_mean_temp', 'ux_err_center_temp', 'uy_err_center_temp']
        for dataname, content in zip(datanames, errdata_list):
            err_data[dataname] = content

        datafilepath = os.path.join(parentdir, 'err_data/' + resultsubdirname)
        rw.write_hdf5_dict(datafilepath, err_data)
    except:
        piv_data.close()
        continue


    piv_data.close()


jhtd_data.close()

print 'Done'