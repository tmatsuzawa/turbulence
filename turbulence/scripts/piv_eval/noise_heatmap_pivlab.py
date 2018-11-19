"""
Compare pivlab outputs with synthetic data for unidirectional velocity fields
... Generates noise pdfs
... Generates heatmaps related to noise
"""

import argparse
import glob
import os
import sys
import numpy as np
from scipy.interpolate import griddata
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import library.tools.rw_data as rw
import library.basics.formatarray as fa
import library.basics.formatstring as fs
import library.basics.std_func as std_func
import library.display.graph as graph
import library.image_processing.movie as movie

parser = argparse.ArgumentParser(description='Make a hdf5 file out of PIVLab txt outputs')
parser.add_argument('-datadir', '--datadir', help='Parent directory of PIVlab output HDF5 directories', type=str,
                    default='/Volumes/labshared3-1/takumi/JHTD-sample/JHT_Database/Data/synthetic_data_unidirectional/hdf5data/')
parser.add_argument('-fdatadir', '--fakedatadir', help='full path to fake data', type=str,
                    default='/Volumes/labshared3-1/takumi/JHTD-sample/JHT_Database/Data/synthetic_data_unidirectional/org_vel_field/')
parser.add_argument('-overwrite', '--overwrite', help='Overwrite output hdf5 file. (Redo fitting)', action='store_true')
parser.add_argument('-mode', '--mode', help='Choose from constant and gradient', type=str,
                    default='constant')



args = parser.parse_args()

# Data architecture
parentdirs = glob.glob(args.datadir + 'W*')
fakedata = glob.glob(args.fakedatadir + 'unidirectional*')
resultdirname = 'comparison_pivlab_org'
datafilename = args.datadir + 'noise_fit_results'
datafilepath = args.datadir + 'noise_fit_results' + '.h5'

# Plotting settings Part 1
cmap = 'RdBu'
cmap2 = 'Blues'
params = {'figure.figsize': (20, 10)
          }
disp_min, disp_max = -0.01, 0.01
graph.update_figure_params(params)

if not os.path.exists(datafilepath) or args.overwrite:
    # Initialize
    disps, iws = [], []
    sigmas_ux, gammas_ux, gauss_peaks_ux, lorentz_peaks_ux = [], [], [], []
    sigmas_ux_err, gammas_ux_err, gauss_peaks_ux_err, lorentz_peaks_ux_err = [], [], [], []
    sigmas_uy, gammas_uy, gauss_peaks_uy, lorentz_peaks_uy = [], [], [], []
    sigmas_uy_err, gammas_uy_err, gauss_peaks_uy_err, lorentz_peaks_uy_err = [], [], [], []

    for parentdir in parentdirs:
        print parentdir
        iw = int(fs.get_float_from_str(parentdir, 'W', 'px')) # interrogation window width
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

                # Find an original velocity field data
                for fakedatum in fakedata:
                    if mag_str in fakedatum:
                        print 'Fake data found!'
                        # update iws and disps for heatmap
                        iws.append(iw)
                        disps.append(mag)

                        # Load fake data
                        fdata = rw.read_hdf5(fakedatum)
                        xx0, yy0 = fdata['x'], fdata['y']
                        ux0, uy0 = fdata['ux'], fdata['uy']
                        # coarse_grain data
                        xx_c = fa.coarse_grain_2darr_overwrap(xx0, iw, iw, overwrap=0.5)
                        yy_c = fa.coarse_grain_2darr_overwrap(yy0, iw, iw, overwrap=0.5)
                        ux_c = fa.coarse_grain_2darr_overwrap(ux0, iw, iw, overwrap=0.5)
                        uy_c = fa.coarse_grain_2darr_overwrap(uy0, iw, iw, overwrap=0.5)

                        ux_err, uy_err = ux - ux_c, uy - uy_c
                        fig1, ax11, bins, hist = graph.pdf(ux_err[np.abs(ux_err) < 0.5], nbins=int(np.sqrt(ux_err.size)*2), return_data=True, fignum=1, subplot=121, label='data')
                        try:
                            fig1, ax11, popt, pcov = graph.plot_fit_curve(bins[(-0.2<bins) * (bins < 0.2)], hist[(-0.2<bins) * (bins < 0.2)],
                                                                           func=std_func.gaussian_norm, fignum=1, subplot=121, label='Gaussian', alpha=0.5)
                            gauss_peaks_ux.append(popt[0])
                            sigmas_ux.append(popt[1])
                            gauss_peaks_ux_err.append(np.sqrt(np.diag(pcov))[0])
                            sigmas_ux_err.append(np.sqrt(np.diag(pcov))[1])
                        except:
                            gauss_peaks_ux.append(np.nan)
                            sigmas_ux.append(np.nan)
                            gauss_peaks_ux_err.append(np.nan)
                            sigmas_ux_err.append(np.nan)
                            pass
                        try:
                            fig1, ax11, popt, pcov = graph.plot_fit_curve(bins[(-0.2<bins) * (bins < 0.2)], hist[(-0.2<bins) * (bins < 0.2)],
                                                                      func=std_func.lorentzian_norm, fignum=1,
                                                                      subplot=121, label='Lorentzian', alpha=0.5)
                            lorentz_peaks_ux.append(popt[0])
                            gammas_ux.append(popt[1])
                            lorentz_peaks_ux_err.append(np.sqrt(np.diag(pcov))[0])
                            gammas_ux_err.append(np.sqrt(np.diag(pcov))[1])
                        except:
                            lorentz_peaks_ux.append(np.nan)
                            gammas_ux.append(np.nan)
                            lorentz_peaks_ux_err.append(np.nan)
                            gammas_ux_err.append(np.nan)
                            pass


                        fig1, ax12, bins, hist = graph.pdf(uy_err[np.abs(uy_err) < 0.5], nbins=int(np.sqrt(uy_err.size)*2), return_data=True, fignum=1, subplot=122, label='data')
                        try:
                            fig1, ax12, popt, pcov = graph.plot_fit_curve(bins[(-0.2<bins) * (bins < 0.2)], hist[(-0.2<bins) * (bins < 0.2)],
                                                                           func=std_func.gaussian_norm, fignum=1, subplot=122, label='Gaussian', alpha=0.5)
                            gauss_peaks_uy.append(popt[0])
                            sigmas_uy.append(popt[1])
                            gauss_peaks_uy_err.append(np.sqrt(np.diag(pcov))[0])
                            sigmas_uy_err.append(np.sqrt(np.diag(pcov))[1])
                        except:
                            gauss_peaks_uy.append(np.nan)
                            sigmas_uy.append(np.nan)
                            gauss_peaks_uy_err.append(np.nan)
                            sigmas_uy_err.append(np.nan)
                            pass
                        try:
                            fig1, ax12, popt, pcov = graph.plot_fit_curve(bins[(-0.2<bins) * (bins < 0.2)], hist[(-0.2<bins) * (bins < 0.2)],
                                                                      func=std_func.lorentzian_norm, fignum=1,
                                                                      subplot=122, label='Lorentzian', alpha=0.5)
                            lorentz_peaks_uy.append(popt[0])
                            gammas_uy.append(popt[1])
                            lorentz_peaks_uy_err.append(np.sqrt(np.diag(pcov))[0])
                            gammas_uy_err.append(np.sqrt(np.diag(pcov))[1])
                        except:
                            lorentz_peaks_uy.append(np.nan)
                            gammas_uy.append(np.nan)
                            lorentz_peaks_uy_err.append(np.nan)
                            gammas_uy_err.append(np.nan)
                            pass


                        # Plotting stuff
                        axes = [ax11, ax12]
                        xlabels, ylabel = [r'$\Delta U_x$ (px/frame)', r'$\Delta U_y$ (px/frame)'], r'Probability density'
                        title = r'W=%dpx, $v$=%.1f px/frame (PIVLab, Original, Diff.)' % (iw, mag)
                        for ax, xlabel in zip(axes, xlabels):
                            graph.labelaxes(ax, xlabel, ylabel)
                            graph.setaxes(ax, -0.2, 0.2, 0, 80)
                            ax.legend()
                        graph.suptitle(title, fignum=1)
                        # fig1.tight_layout()

                        # Close hdf5 files
                        pivdata.close()
                        fdata.close()

                        figname = '/noisedist/pivlab_fakedata_comp_W%d_v%spxframe_multiple_passes' % (iw, mag_str_2)
                        graph.save(resultdir + figname)
                        # plt.show()
                        plt.close()
                        break
        #     elif args.mode == 'gradient':
        #         max = fs.get_float_from_str(pivdatum_loc, 'max', '_min')  # max velocity
        #         min = fs.get_float_from_str(pivdatum_loc, 'min', '.h5')  # min velocity
        #
        #         max_str = 'max' + fs.convert_float_to_decimalstr(max)
        #         min_str = 'min' + fs.convert_float_to_decimalstr(min)
        #         max_str_2 = 'max' + fs.convert_float_to_decimalstr(max, zpad=4) # for naming scheme
        #         min_str_2 = 'min' + fs.convert_float_to_decimalstr(min, zpad=4) # for naming scheme
        #
        #         # Plotting settings Part 2
        #         vmin, vmax = min, max * 1.2
        #
        #         # Load pivlab output
        #         pivdata = rw.read_hdf5(pivdatum_loc)
        #         xx, yy = pivdata['x'], pivdata['y']
        #         ux, uy = pivdata['ux'][..., 0], pivdata['uy'][..., 0]
        #
        #         fig1, ax11, cc11 = graph.color_plot(xx, yy, ux, vmin=vmin, vmax=vmax, cmap=cmap2, fignum=1, subplot=231)
        #         fig1, ax14, cc14 = graph.color_plot(xx, yy, uy, vmin=-1, vmax=1, cmap=cmap, fignum=1, subplot=234)
        #
        #         # Find an original velocity field data
        #         for fakedatum in fakedata:
        #             if max_str in fakedatum and min_str in fakedatum:
        #                 print 'Fake data found!'
        #                 # Load fake data
        #                 fdata = rw.read_hdf5(fakedatum)
        #                 xx0, yy0 = fdata['x'], fdata['y']
        #                 ux0, uy0 = fdata['ux'], fdata['uy']
        #                 fig1, ax12, cc12 = graph.color_plot(xx0, yy0, ux0, vmin=vmin, vmax=vmax, cmap=cmap2, fignum=1,
        #                                                     subplot=232)
        #                 fig1, ax15, cc15 = graph.color_plot(xx0, yy0, uy0, vmin=-1, vmax=1, cmap=cmap, fignum=1,
        #                                                     subplot=235)
        #                 # coarse_grain data
        #                 xx_c = fa.coarse_grain_2darr_overwrap(xx0, iw, iw, overwrap=0.5)
        #                 yy_c = fa.coarse_grain_2darr_overwrap(yy0, iw, iw, overwrap=0.5)
        #                 ux_c = fa.coarse_grain_2darr_overwrap(ux0, iw, iw, overwrap=0.5)
        #                 uy_c = fa.coarse_grain_2darr_overwrap(uy0, iw, iw, overwrap=0.5)
        #                 fig1, ax13, cc13 = graph.color_plot(xx, yy, ux - ux_c, vmin=-1, vmax=1, cmap=cmap, fignum=1,
        #                                                     subplot=233)
        #                 fig1, ax16, cc16 = graph.color_plot(xx, yy, uy - uy_c, vmin=-1, vmax=1, cmap=cmap, fignum=1,
        #                                                     subplot=236)
        #
        #                 # Plotting stuff
        #                 axes = [ax11, ax12, ax13, ax14, ax15, ax16]
        #                 ccs = [cc11, cc12, cc13, cc14, cc15, cc16]
        #                 cblabels = [r'$U_x$', r'$U_x$', r'$\Delta U_x$', r'$U_y$', r'$U_y$', r'$\Delta U_y$']
        #                 xlabel, ylabel = r'$X$ (px)', r'$Y$ (px)'
        #                 title = r'W=%dpx, $v_{max}$=%.1f px/frame (PIVLab, Original, Diff.)' % (iw, max)
        #                 for ax, cc, cblabel in zip(axes, ccs, cblabels):
        #                     graph.add_colorbar(cc, ax=ax, label=cblabel)
        #                     graph.labelaxes(ax, xlabel, ylabel)
        #                     graph.setaxes(ax, 0, xx0.shape[1], 0, xx0.shape[0])
        #                 graph.suptitle(title, fignum=1)
        #                 fig1.tight_layout()
        #
        #                 pivdata.close()
        #                 fdata.close()
        #
        #                 figname = '/pivlab_fakedata_comp_W%d_vmax%spxframe_vmin%spxframe_multiple_passes' % (iw, max_str_2, min_str_2)
        #                 graph.save(resultdir + figname)
        #                 plt.close()
        #                 break
        #
        # imgfiles = resultdir + figname[:6]
        # moviename = '/movie_W%dpx_multiple_passes' % iw
        # movie.make_movie(imgfiles, resultdir + moviename, framerate=1, option='glob')

    # save data
    data = {}
    data['inter_wid_width'] = iws
    data['displacement'] = disps
    data['sigma_ux'] = sigmas_ux
    data['sigma_uy'] = sigmas_uy
    data['mu_ux_gauss'] = gauss_peaks_ux
    data['mu_uy_gauss'] = gauss_peaks_uy
    data['gamma_ux'] = gammas_ux
    data['gamma_uy'] = gammas_uy
    data['mu_ux_lorentz'] = lorentz_peaks_ux
    data['mu_uy_lorentz'] = lorentz_peaks_uy
    data['sigma_ux_err'] = sigmas_ux_err
    data['sigma_uy_err'] = sigmas_uy_err
    data['mu_ux_gauss_err'] = gauss_peaks_ux_err
    data['mu_uy_gauss_err'] = gauss_peaks_uy_err
    data['gamma_ux_err'] = gammas_ux_err
    data['gamma_uy_err'] = gammas_uy_err
    data['mu_ux_lorentz_err'] = lorentz_peaks_ux_err
    data['mu_uy_lorentz_err'] = lorentz_peaks_uy_err
    rw.write_hdf5_dict(datafilename, data)

else:
    print 'Noise fit results already exist!'
    data = rw.read_hdf5(datafilepath)
    iws, disps = np.array(data['inter_wid_width']), np.array(data['displacement'])
    sigmas_ux, sigmas_uy, gauss_peaks_ux, gauss_peaks_uy = np.array(data['sigma_ux']), np.array(data['sigma_uy']), np.array(data['mu_ux_gauss']), np.array(data['mu_uy_gauss'])
    gammas_ux, gammas_uy, lorentz_peaks_ux, lorentz_peaks_uy = np.array(data['gamma_ux']), np.array(data['gamma_uy']), np.array(data['mu_ux_lorentz']), np.array(data['mu_uy_lorentz'])
    sigmas_ux_err, sigmas_uy_err, gauss_peaks_ux_err, gauss_peaks_uy_err = np.array(data['sigma_ux_err']), np.array(data['sigma_uy_err']), np.array(data['mu_ux_gauss_err']), np.array(data['mu_uy_gauss_err'])
    gammas_ux_err, gammas_uy_err, lorentz_peaks_ux_err, lorentz_peaks_uy_err = np.array(data['gamma_ux_err']), np.array(data['gamma_uy_err']), np.array(data['mu_ux_lorentz_err']), np.array(data['mu_uy_lorentz_err'])


# Now make a heat map
iw_temp, disp_temp = np.linspace(0, 80, 81), np.linspace(0, 80, 81)
iw_grid, disp_grid = np.meshgrid(iw_temp, disp_temp)
sigma_ux_int = griddata(np.array(zip(iws, disps)), np.abs(np.array(sigmas_ux)), (iw_grid, disp_grid), method='nearest')
sigma_uy_int = griddata(np.array(zip(iws, disps)), np.abs(np.array(sigmas_uy)), (iw_grid, disp_grid), method='nearest')
mu_ux_gauss_int = griddata(np.array(zip(iws, disps)), np.array(gauss_peaks_ux), (iw_grid, disp_grid), method='nearest')
mu_uy_gauss_int = griddata(np.array(zip(iws, disps)), np.array(gauss_peaks_uy), (iw_grid, disp_grid), method='nearest')
gamma_ux_int = griddata(np.array(zip(iws, disps)), np.array(gammas_ux), (iw_grid, disp_grid), method='nearest')
gamma_uy_int = griddata(np.array(zip(iws, disps)), np.array(gammas_uy), (iw_grid, disp_grid), method='nearest')
mu_ux_lorentz_int = griddata(np.array(zip(iws, disps)), np.array(lorentz_peaks_ux), (iw_grid, disp_grid), method='nearest')
mu_uy_lorentz_int = griddata(np.array(zip(iws, disps)), np.array(lorentz_peaks_uy), (iw_grid, disp_grid), method='nearest')

axes, ccs = [], []
data2plot = [mu_ux_gauss_int, sigma_ux_int, mu_uy_gauss_int, sigma_uy_int, mu_ux_lorentz_int, gamma_ux_int, mu_uy_lorentz_int, gamma_uy_int]
cmaps = [cmap, cmap2, cmap, cmap2, cmap, cmap2, cmap, cmap2]
vmins = [disp_min/2, 0, disp_min/2, 0, disp_min/2, 0, disp_min/2, 0]
vmaxs = [disp_max/2, disp_max*2, disp_max/2, disp_max*2, disp_max/2, disp_max*2, disp_max/2, disp_max*2]
titles = [r'$\mu_{\Delta x}$', r'$\sigma_{\Delta x}$', r'$\mu_{\Delta y}$', r'$\sigma_{\Delta y}$',
          '$\chi_{\Delta x}$', r'$\gamma_{\Delta x}$', r'$\chi_{\Delta y}$', r'$\gamma_{\Delta y}$']
xlabel, ylabel = r'Interrogation window size (px)', r'$\Delta x$ (px)'
for i, fitdata in enumerate(data2plot):
    fig2, ax2, cc2 = graph.color_plot(iw_grid, disp_grid, fitdata, vmin=vmins[i], vmax=vmaxs[i], fignum=2, subplot=241+i, cmap=cmaps[i])
    plt.scatter(iws, disps, color='m', s=4)
    axes.append(ax2)
    ccs.append(cc2)

def vel_upper_limit_func(iw, npass):
    iw_ini = iw * 2**(npass-1)
    return iw_ini/2

for ax, title, cc in zip(axes, titles, ccs):
    ax.plot(np.linspace(0, 80), vel_upper_limit_func(np.linspace(0, 80), 4), linestyle='--', color='r')
    ax.set_facecolor('k')
    graph.setaxes(ax, 0, 80, 0, 80)
    graph.add_colorbar(cc, ax=ax)
    graph.labelaxes(ax, xlabel, ylabel)
    graph.title(ax, title)
graph.save(datafilename+'4')


#sort arrays first
import library.basics.formatarray as fa
iws_s, disps_s = fa.sort_two_arrays_using_order_of_first_array(iws, disps)
# Accuracy
iws_s, gauss_peaks_ux_s = fa.sort_two_arrays_using_order_of_first_array(iws, gauss_peaks_ux)
iws_s, gauss_peaks_ux_err_s = fa.sort_two_arrays_using_order_of_first_array(iws, gauss_peaks_ux_err)
iws_s, lorentz_peaks_ux_s = fa.sort_two_arrays_using_order_of_first_array(iws, lorentz_peaks_ux)
iws_s, lorentz_peaks_ux_err_s = fa.sort_two_arrays_using_order_of_first_array(iws, lorentz_peaks_ux_err)
# Precision
iws_s, sigmas_ux_s = fa.sort_two_arrays_using_order_of_first_array(iws, sigmas_ux)
iws_s, sigmas_ux_err_s = fa.sort_two_arrays_using_order_of_first_array(iws, sigmas_ux_err)
iws_s, gammas_ux_s = fa.sort_two_arrays_using_order_of_first_array(iws, gammas_ux)
iws_s, gammas_ux_err_s = fa.sort_two_arrays_using_order_of_first_array(iws, gammas_ux_err)


labels = [ r'W=8px', r'W=16px', r'W=32px', r'W=64px']
for i in range(4):
    fig3, ax31 = graph.errorbar(disps_s[i*6:(i+1)*6], gauss_peaks_ux_s[i*6:(i+1)*6], yerr=gauss_peaks_ux_err_s[i*6:(i+1)*6], fignum=3, subplot=221, label=labels[i])
    fig3, ax32 = graph.errorbar(disps_s[i * 6:(i + 1) * 6], lorentz_peaks_ux_s[i*6:(i+1)*6], yerr=lorentz_peaks_ux_err_s[i*6:(i+1)*6], fignum=3,
                                subplot=222, label=labels[i])
    fig3, ax33 = graph.errorbar(disps_s[i*6:(i+1)*6], np.abs(sigmas_ux_s[i*6:(i+1)*6]), yerr=sigmas_ux_err_s[i*6:(i+1)*6], fignum=3, subplot=223, label=labels[i])
    fig3, ax34 = graph.errorbar(disps_s[i * 6:(i + 1) * 6], gammas_ux_s[i*6:(i+1)*6], yerr=gammas_ux_err_s[i*6:(i+1)*6], fignum=3,
                                subplot=224, label=labels[i])

axes3 = [ax31, ax32, ax33, ax34]
xlabel, ylabels = r'$\Delta x$ (px)', [r'$\mu_{\Delta x}$ (px)', r'$\chi_{\Delta x}$ (px)', r'$\sigma_{\Delta x}$ (px)', r'$\gamma_{\Delta x}$ (px)']
ymins = [-0.01, -0.01, 0, 0]
ymaxs = [0.01, 0.01, 0.02, 0.02]
for i, ax in enumerate(axes3):
    graph.labelaxes(ax, xlabel, ylabels[i])
    graph.setaxes(ax, 0, 68, ymins[i], ymaxs[i])
    ax.legend(loc=1)
graph.save(datafilename+'_5')
graph.show()



# close hdf5 file if it was used
if os.path.exists(datafilepath) and not args.overwrite:
        data.close()

