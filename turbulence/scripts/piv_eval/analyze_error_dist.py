import numpy as np
import h5py
import glob
import library.basics.formatstring as fs
import library.basics.formatarray as fa
import library.basics.std_func as std_func
import library.image_processing.movie as movie
import library.tools.rw_data as rw
import library.display.graph as graph
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
import os
import copy

overwrite = True

def search_tuple(tups, elem):
    """
    Searches an element in tuple, and returns a list of found tuple elements

    Example:
    tuples = [(1, "hey"), (2, "hey"), (3, "no")]
    print(search_tuple(tuples, "hey"))
    >> [(1, "hey"), (2, "hey")]
    print(search_tuple(tuples, 3))
    >> [(3, "no")]

    Parameters
    ----------
    tups
    elem

    Returns
    -------

    """
    return filter(lambda tup: elem in tup, tups)

# Data architecture
# dir = '/Volumes/bigraid/takumi/turbulence/JHTD/synthetic_data/hdf5data/tstep1_npt50000_lt20p0_pbcTrue_varyingDt/post_processed/err_data'
dir = '/Volumes/bigraid/takumi/turbulence/JHTD/synthetic_data/hdf5data/tstep1_npt50000_lt20p0_pbcTrue_no_uz_varyingDt/post_processed/err_data'
errdata_list = glob.glob(dir + '/*.h5')
errdata_list = fa.natural_sort(errdata_list)

# jhtd parameters
# dx = 2 * np.pi / 1024. # unit length / px
dy = dz = dx = 1  # in px
dt_sim = 0.0002  # DNS simulation time step

dt_spacing = 10
# dt = dt_sim * param['tstep'] * dt_spacing # time separation between data points in JHTD time unit
nu = 0.000185  # viscosity (unit length ^2 / unit time)
fx = (2 * np.pi) / 1024  # unit length / px
tau = 0.19 # time scale of out-of-plane motion
taueta = 0.0424 # Kolmogorov timescale in a.u.

# Plotting settings 1
params = {'figure.figsize': (20, 10),
          'font.size': 20,
          'legend.fontsize': 10,
          'axes.labelsize': 20}
lw = 4 # line width of plots
graph.update_figure_params(params)
cmap = mpl.cm.get_cmap('magma')
cmap2 = 'RdBu'
normalize = mpl.colors.Normalize(vmin=-512, vmax=512)

# Parameters for analysis
ustep = 10  # range of u for error analysis
u0, u1 = -550, 550 # min and max of u for error analysis
umins = range(u0, u1 - ustep, ustep)
umaxes = range(u0 + ustep, u1, ustep)

# initialize fitdata array
fitdata_arr = np.empty((len(umins), 9, 7, 4)) # (uxtrue, deltat, fitparams, W)
fitdata_arr[:] = np.nan

ux_points = np.empty((len(umins), 9))
ux_points[:] = np.nan
deltat_points = np.empty((len(umins), 9))
deltat_points[:] = np.nan
#
# for i, errdata_path in enumerate(errdata_list):
#     # if i>0:
#     #     break
#     print errdata_path
#     errdata = h5py.File(errdata_path, mode='r')
#     resultname = fs.get_filename_wo_ext(errdata_path)
#
#     keys_all = [u'ux_center_temp', u'ux_err_center_temp', u'ux_err_mean_temp', u'ux_mean',
#                 u'uy_center_temp', u'uy_err_center_temp', u'uy_err_mean_temp', u'uy_mean']
#
#     keys = [u'ux_mean',  u'ux_err_mean_temp', u'uy_mean', u'uy_err_mean_temp']
#     titles = [r'$W=8$px', '$W=16$px', '$W=32$px', '$W=64$px']
#     suptitles = [r'$\Delta U_x$', r'$\Delta U_{x,center}$', r'$\Delta U_y$', r'$\Delta U_{y,center}$']
#     iws = [8, 16, 32, 64]
#     subplots = [221, 222, 223, 224]
#     subplot_tpl = zip(iws, subplots)
#
#
#     deltat = fs.get_float_from_str(errdata_path, 'Dt_',
#                                    '_')  # number of DNS steps between image A and image B = deltat * 10 for isotropic1024coarse
#     iw = fs.get_float_from_str(errdata_path, 'W', 'pix')  # interrogation window size in px
#     lt = fs.get_float_from_str(errdata_path, 'lt', '_')  # laser thickness in px
#
#
#     # Plotting settings 2
#     subplot = search_tuple(subplot_tpl, iw)[0][1]  # 3 digit subplot index obtained from subplot_tpl
#     # label = '$\Delta t$=%d DNS steps = %.3f (a.u)' % (deltat * dt_spacing, deltat * dt_spacing * dt_sim)
#     label = '$\Delta t = %.3f $ (a.u)' % (deltat * dt_spacing * dt_sim)
#
#     vmax = 1024 / 2 * dt_sim * (deltat * dt_spacing)  # px/frame
#     # vmax = iw * 8 / 2
#     vmin = - vmax
#     ft = deltat * dt_sim * dt_spacing  # conversion factor (time unit / frame)
#
#     counter = 0
#     for j, (umin, umax) in enumerate(zip(umins, umaxes)):
#         print 'Analyzing data with a velocity range [umin, umax): [%d, %d) ' % (umin, umax)
#         ux_points[j, int(deltat / 10 - 1)] = float(umin + umax) / 2.  # px / time unit
#         deltat_points[j, int(deltat / 10 - 1)] = deltat * dt_sim * dt_spacing  # time unit
#
#         label=r'$%.1f \leq U_i < %.1f$ (px/unit time)' % (umin, umax)
#
#         # Get error characteristics for velocity beween umin and umax
#         ind1 = np.asarray(np.asarray(errdata[keys[0]]) / ft) > umin # px / time unit
#         ind2 = np.asarray(np.asarray(errdata[keys[0]]) / ft) <= umax # px / time unit
#         ind = ind1 * ind2
#         vdata_small = np.asarray(errdata[keys[0]])[ind] / ft # px/ unit time
#         errdata_small = np.asarray(errdata[keys[1]])[ind]/ ft # px / unit time
#         if len(errdata_small) < 100:
#             print 'Too few data points to take statistics! Skipping a velocity range [umin, umax): [%d, %d) ' % (umin, umax)
#             continue
#         counter += 1
#         # Velocity dependence on error
#         fig1, ax1, bins1, hist1 = graph.pdf(errdata_small, nbins=int(np.sqrt(len(errdata_small))), fignum=1, label=label, alpha=1, lw=6, color='C1', return_data=True)
#
#
#         # ... make a small scatter plot inside a main plot
#         # subax1 = graph.add_subplot_axes(ax1, [0.05, 0.55, 0.4, 0.4])
#         # subax1.scatter(np.asarray(errdata[keys[0]])[~ind] / ft, np.asarray(errdata[keys[1]])[~ind] / ft, alpha=0.05)
#         # subax1.scatter(np.asarray(errdata[keys[0]])[ind] / ft, np.asarray(errdata[keys[1]])[ind] / ft, color='C1', alpha=0.05)
#         # subax1.set_xlabel(r'$U_x^{true}$ (px/unit time)', fontsize=10)
#         # subax1.set_ylabel(r'$\Delta U_x$ (px/unit time)', fontsize=10)
#         # graph.axvband(subax1, umin, umax, color='C1')
#
#         # ax1.legend()
#         ax1.set_xlim(vmin / ft, vmax / ft)
#         ax1.set_ylim(0, 0.04)
#         graph.labelaxes(ax1, r'$\Delta U_x$ (px/unit time)', 'Probabilistic density')
#         graph.title(ax1, label)
#         graph.suptitle(r'W=%dpx, $\Delta t = 1$ frame $= %.3f$ (a.u.) $= %d$ DNS steps = $ %.2f \tau_{\eta}$' % (iw, deltat * dt_sim * dt_spacing, deltat * dt_spacing, deltat * dt_sim * dt_spacing / taueta))
#         # save
#         imgpath1 = dir + '/results/W%dpx_vel_selected/Dt_%02d/err_pdf_%03d_' % (iw, deltat, counter) + resultname + 'umin_%d_umax_%d_' %(umin, umax)
#         graph.save(imgpath1, ext='png', fignum=1)
#
#         # Velocity dependence on error (ALL)
#         fig5, ax5 = graph.pdf(errdata_small, nbins=int(np.sqrt(len(errdata_small))), fignum=5, label=label, color=cmap(normalize((umin+umax)/2)), alpha=0.5, lw=6)
#         ax5.legend()
#         ax5.set_xlim(vmin / ft, vmax / ft)
#         graph.labelaxes(ax5, r'$\Delta U_x(\vec{x}) / U_x^{true}(\vec{x})$', 'Probabilistic density')
#         graph.suptitle(r'W=%dpx, $\Delta t = 1$ frame $= %.3f$ (a.u.) $= %d$ DNS steps = $ %.2f \tau_{\eta}$' % (iw, deltat * dt_sim * dt_spacing, deltat * dt_spacing, deltat * dt_sim * dt_spacing / taueta))
#
#         # Velocity dependence on error ratio
#         fig6, ax6 = graph.pdf(errdata_small / vdata_small, nbins=int(np.sqrt(len(errdata_small))), fignum=6, label=label, alpha=1, lw=6, color='C1')
#         # ... make a small scatter plot inside a main plot
#         subax6 = graph.add_subplot_axes(ax6, [0.05, 0.6, 0.4, 0.4])
#         subax6.scatter(np.asarray(errdata[keys[0]])[~ind] / ft, np.asarray(errdata[keys[1]])[~ind] / ft, alpha=0.05)
#         subax6.scatter(np.asarray(errdata[keys[0]])[ind] / ft, np.asarray(errdata[keys[1]])[ind] / ft, color='C1', alpha=0.05)
#         subax6.set_xlabel(r'$U_x^{true}$ (px/unit time)', fontsize=10)
#         subax6.set_ylabel(r'$\Delta U_x$ (px/unit time)', fontsize=10)
#         graph.axvband(subax6, umin, umax, color='C1')
#         ax6.set_xlim(-5, 5)
#         ax6.set_ylim(0, 4)
#         graph.labelaxes(ax6, r'$\Delta U_x(\vec{x}) / U_x^{true}(\vec{x})$', 'Probabilistic density')
#         graph.title(ax6, label)
#         graph.suptitle(r'W=%dpx, $\Delta t = 1$ frame $= %.3f$ (a.u.) $= %d$ DNS steps = $ %.2f \tau_{\eta}$' % (
#         iw, deltat * dt_sim * dt_spacing, deltat * dt_spacing, deltat * dt_sim * dt_spacing / taueta))
#         # save
#         imgpath6 = dir + '/results/W%dpx_vel_selected_ratio/Dt_%02d/err_pdf_%03d_' % (
#         iw, deltat, counter) + resultname + 'umin_%d_umax_%d_' % (umin, umax)
#         graph.save(imgpath6, ext='png', fignum=6)
#
#         # Velocity dependence on error ratio (ALL)
#         fig7, ax7 = graph.pdf(errdata_small / vdata_small, nbins=int(np.sqrt(len(errdata_small))), fignum=7, label=label, color=cmap(normalize((umin+umax)/2)), alpha=0.5, lw=6)
#         ax7.legend()
#         ax7.set_xlim(-5, 5)
#         graph.labelaxes(ax7, r'$\Delta U_x$ (px/unit time)', 'Probabilistic density')
#         graph.suptitle(r'W=%dpx, $\Delta t = 1$ frame $= %.3f$ (a.u.) $= %d$ DNS steps = $ %.2f \tau_{\eta}$' % (iw, deltat * dt_sim * dt_spacing, deltat * dt_spacing, deltat * dt_sim * dt_spacing / taueta))
#
#         # plt.close(fig1)
#         plt.close(fig6)
#
#         # fit error distribution to lorentzian
#         try:
#             # use points neare peak for fitting
#             # qmin, qmax = np.min(hist1), np.max(hist1)
#             # ind1 = hist1 > (qmin + (qmax - qmin)/5.)
#             # bins1, hist1 = bins1[ind1], hist1[ind1]
#
#             popt, pcov = curve_fit(std_func.lorentzian, bins1, hist1)
#             fit_err = np.sqrt(np.diag(pcov))
#             y, y_fit = hist1, std_func.lorentzian(bins1, *popt)
#             ss_res = np.sum((y - y_fit) ** 2)
#             ss_tot = np.sum((y - np.nanmean(y)) ** 2)
#             r2     = 1 - (ss_res / ss_tot)
#
#             # draw fit curve
#             graph.plot_fit_curve(bins1, hist1, std_func.lorentzian, fignum=1)
#             subax1 = graph.add_subplot_axes(ax1, [0.1, 0.55, 0.4, 0.4])
#             subax1.scatter(np.asarray(errdata[keys[0]])[~ind] / ft, np.asarray(errdata[keys[1]])[~ind] / ft, alpha=0.05)
#             subax1.scatter(np.asarray(errdata[keys[0]])[ind] / ft, np.asarray(errdata[keys[1]])[ind] / ft, color='C1', alpha=0.005)
#             subax1.set_xlabel(r'$U_x^{true}$ (px/unit time)', fontsize=10)
#             subax1.set_ylabel(r'$\Delta U_x$ (px/unit time)', fontsize=10)
#             subax1.set_xlim(vmin / ft, vmax / ft)
#             subax1.set_ylim(vmin / ft*2, vmax / ft*2)
#             graph.axvband(subax1, umin, umax, color='C1')
#
#             # ax1.legend()
#             ax1.set_xlim(vmin / ft, vmax / ft)
#             ax1.set_ylim(0, 0.04)
#             graph.labelaxes(ax1, r'$\Delta U_x$ (px/unit time)', 'Probabilistic density')
#             graph.title(ax1, label)
#             graph.suptitle(r'W=%dpx, $\Delta t = 1$ frame $= %.3f$ (a.u.) $= %d$ DNS steps = $ %.2f \tau_{\eta}$' % (iw, deltat * dt_sim * dt_spacing, deltat * dt_spacing, deltat * dt_sim * dt_spacing / taueta))
#             # save
#             imgpath1 = dir + '/results/W%dpx_vel_selected/Dt_%02d/err_pdf_%03d_' % (iw, deltat, counter) + resultname + 'umin_%d_umax_%d_' %(umin, umax)
#             graph.save(imgpath1, ext='png', fignum=1)
#             plt.close(fig1)
#
#             fitdata = [popt[0], np.abs(popt[1]), popt[2], fit_err[0], fit_err[1], fit_err[2], r2]
#             for k in range(fitdata_arr.shape[2]): # (uxtrue, deltat, fitparams, W)
#                 print fitdata[k]
#                 fitdata_arr[j, int(deltat/10-1), k, int(np.log2(iw/8))] = fitdata[k]
#         except:
#             subax1 = graph.add_subplot_axes(ax1, [0.1, 0.55, 0.4, 0.4])
#             subax1.scatter(np.asarray(errdata[keys[0]])[~ind] / ft, np.asarray(errdata[keys[1]])[~ind] / ft, alpha=0.05)
#             subax1.scatter(np.asarray(errdata[keys[0]])[ind] / ft, np.asarray(errdata[keys[1]])[ind] / ft, color='C1', alpha=0.005)
#             subax1.set_xlabel(r'$U_x^{true}$ (px/unit time)', fontsize=10)
#             subax1.set_ylabel(r'$\Delta U_x$ (px/unit time)', fontsize=10)
#             subax1.set_ylim(vmin / ft, vmax / ft)
#             subax1.set_ylim(vmin / ft*2, vmax / ft*2)
#             graph.axvband(subax1, umin, umax, color='C1')
#
#             # ax1.legend()
#             ax1.set_xlim(vmin / ft, vmax / ft)
#             ax1.set_ylim(0, 0.04)
#             graph.labelaxes(ax1, r'$\Delta U_x$ (px/unit time)', 'Probabilistic density')
#             graph.title(ax1, label)
#             graph.suptitle(r'W=%dpx, $\Delta t = 1$ frame $= %.3f$ (a.u.) $= %d$ DNS steps = $ %.2f \tau_{\eta}$' % (iw, deltat * dt_sim * dt_spacing, deltat * dt_spacing, deltat * dt_sim * dt_spacing / taueta))
#             # save
#             imgpath1 = dir + '/results/W%dpx_vel_selected/Dt_%02d/err_pdf_%03d_' % (iw, deltat, counter) + resultname + 'umin_%d_umax_%d_' %(umin, umax)
#             graph.save(imgpath1, ext='png', fignum=1)
#             plt.close(fig1)
#
#             print 'Fitting error... skip'
#             continue
#
#     imgpath5 = dir + '/results/W%dpx_vel_selected/all/all_Ux_Dt%d_' % (iw, deltat)
#     graph.save(imgpath5, ext='png', fignum=5)
#     plt.close(fig5)
#     imgpath7 = dir + '/results/W%dpx_vel_selected_ratio/all/all_Ux_Dt%d_' % (iw, deltat)
#     graph.save(imgpath7, ext='png', fignum=7)
#     plt.close(fig7)
# #
#
#
#     # Error distribution: Ux-DeltaUx and Uy-DeltaUy (px/frame)
#     fig2, ax21 = graph.scatter(np.asarray(errdata[keys[0]]), np.asarray(errdata[keys[1]]), alpha=0.05, fignum=2, subplot=121)
#     fig2, ax22 = graph.scatter(np.asarray(errdata[keys[2]]), np.asarray(errdata[keys[3]]), alpha=0.05, fignum=2, subplot=122)
#     graph.setaxes(ax21, vmin, vmax, vmin, vmax)
#     graph.setaxes(ax22, vmin, vmax, vmin, vmax)
#     graph.labelaxes(ax21, r'$U_x^{true}$ (px/frame)', r'$\Delta U_x = U_x^{piv} - U_x^{true}$ (px/frame)')
#     graph.labelaxes(ax22, r'$U_y^{true}$ (px/frame)', r'$\Delta U_y = U_y^{piv} - U_y^{true}$ (px/frame)')
#     graph.suptitle(r'$\Delta t = 1$ frame $= %.3f$ (a.u.) $= %d$ DNS steps = $ %.2f \tau_{\eta}$' % (deltat * dt_sim * dt_spacing, deltat * dt_spacing, deltat * dt_sim * dt_spacing / taueta))
#
#     # Error distribution: Ux-DeltaUx and Uy-DeltaUy (px/time unit)
#     fig3, ax31 = graph.scatter(np.asarray(errdata[keys[0]]) / ft, np.asarray(errdata[keys[1]]) / ft, alpha=0.05, fignum=3, subplot=121)
#     fig3, ax32 = graph.scatter(np.asarray(errdata[keys[2]]) / ft, np.asarray(errdata[keys[3]]) / ft, alpha=0.05, fignum=3, subplot=122)
#     graph.setaxes(ax31, vmin / ft, vmax / ft, vmin / ft, vmax / ft)
#     graph.setaxes(ax32, vmin / ft, vmax / ft, vmin / ft, vmax / ft)
#     graph.labelaxes(ax31, r'$U_x^{true}$ (px/unit time)', r'$\Delta U_x = U_x^{piv} - U_x^{true}$ (px/unit time)')
#     graph.labelaxes(ax32, r'$U_y^{true}$ (px/unit time)', r'$\Delta U_y = U_y^{piv} - U_y^{true}$ (px/unit time)')
#     graph.suptitle(r'$\Delta t = 1$ frame $= %.3f$ (a.u.) $= %d$ DNS steps = $ %.2f \tau_{\eta}$' % (
#     deltat * dt_sim * dt_spacing, deltat * dt_spacing, deltat * dt_sim * dt_spacing / taueta))
#
#     # Error distribution: Ux-DeltaUx/Ux and Uy-DeltaUy/Uy (px/time unit)
#     ft = deltat * dt_sim * dt_spacing # conversion factor (time unit / frame)
#     fig4, ax41 = graph.scatter(np.asarray(errdata[keys[0]]) / ft, np.asarray(errdata[keys[1]]) / np.asarray(errdata[keys[0]]), alpha=0.05, fignum=4, subplot=121)
#     fig4, ax42 = graph.scatter(np.asarray(errdata[keys[2]]) / ft, np.asarray(errdata[keys[3]]) / np.asarray(errdata[keys[2]]), alpha=0.05, fignum=4, subplot=122)
#     graph.setaxes(ax41, vmin / ft, vmax / ft, -5, 5)
#     graph.setaxes(ax42, vmin / ft, vmax / ft, -5, 5)
#     graph.labelaxes(ax41, r'$U_x^{true}$ (px/unit time)', r'$\Delta U_x / U_x^{true}$')
#     graph.labelaxes(ax42, r'$U_y^{true}$ (px/unit time)', r'$\Delta U_y / U_y^{true}$')
#     graph.suptitle(r'$\Delta t = 1$ frame $= %.3f$ (a.u.) $= %d$ DNS steps = $ %.2f \tau_{\eta}$' % (
#     deltat * dt_sim * dt_spacing, deltat * dt_spacing, deltat * dt_sim * dt_spacing / taueta))
#
#
#     imgpath2 = dir + '/results/W%dpx/err_dist_' % iw + resultname
#     imgpath3 = dir + '/results/W%dpx_converted/err_dist_' % iw + resultname
#     imgpath4 = dir + '/results/W%dpx_converted_ratio_zoomed/err_dist_' % iw + resultname
#     graph.save(imgpath2, fignum=2, ext='png')
#     graph.save(imgpath3, fignum=3, ext='png')
#     graph.save(imgpath4, fignum=4, ext='png')
#
#
#     plt.close('all')
#
#     errdata.close()
#
#
#
#
#
#
# # make movies
# moviedirs = [dir + '/results/W8px', dir + '/results/W16px', dir + '/results/W32px', dir + '/results/W64px',
#              dir + '/results/W8px_converted', dir + '/results/W16px_converted',
#              dir + '/results/W32px_converted', dir +'/results/W64px_converted',
#              dir + '/results/W8px_converted_ratio_zoomed', dir + '/results/W16px_converted_ratio_zoomed',
#              dir + '/results/W32px_converted_ratio_zoomed', dir + '/results/W64px_converted_ratio_zoomed',
#              dir + '/results/W8dpx_vel_selected', dir + '/results/W16dpx_vel_selected',
#              dir + '/results/W32dpx_vel_selected', dir + '/results/W32dpx_vel_selected']
# moredirs1 = [dir + '/results/W8px_vel_selected/Dt_' + str(tt) for tt in range(10, 100, 10)]
# moredirs2 = [dir + '/results/W16px_vel_selected/Dt_' + str(tt) for tt in range(10, 100, 10)]
# moredirs3 = [dir + '/results/W32px_vel_selected/Dt_' + str(tt) for tt in range(10, 100, 10)]
# moredirs4 = [dir + '/results/W64px_vel_selected/Dt_' + str(tt) for tt in range(10, 100, 10)]
# moredirs5 = [dir + '/results/W8px_vel_selected_ratio/Dt_' + str(tt) for tt in range(10, 100, 10)]
# moredirs6 = [dir + '/results/W16px_vel_selected_ratio/Dt_' + str(tt) for tt in range(10, 100, 10)]
# moredirs7 = [dir + '/results/W32px_vel_selected_ratio/Dt_' + str(tt) for tt in range(10, 100, 10)]
# moredirs8 = [dir + '/results/W64px_vel_selected_ratio/Dt_' + str(tt) for tt in range(10, 100, 10)]
# moviedirs += moredirs1 + moredirs2 + moredirs3 + moredirs4 + moredirs5 + moredirs6 + moredirs7 + moredirs8
# # moviedirs = moredirs1 + moredirs2 + moredirs3 + moredirs4
#
# for moviedir in moviedirs:
#     movie. make_movie(moviedir, framerate=3, rm_images=False, ext='png', option='glob', overwrite=True)
#
#
#

### Apparently, storing fitting results leads to a segmentation error for attempting to use restricted memory.
# I do not understand this well. Instead, store data into pickles first
# Read
pklpath1 = '/Volumes/bigraid/takumi/turbulence/JHTD/synthetic_data/hdf5data/tstep1_npt50000_lt20p0_pbcTrue_no_uz_varyingDt/post_processed/err_data/fitdata.pkl'
pklpath2 = '/Volumes/bigraid/takumi/turbulence/JHTD/synthetic_data/hdf5data/tstep1_npt50000_lt20p0_pbcTrue_no_uz_varyingDt/post_processed/err_data/ux_points.pkl'
pklpath3 = '/Volumes/bigraid/takumi/turbulence/JHTD/synthetic_data/hdf5data/tstep1_npt50000_lt20p0_pbcTrue_no_uz_varyingDt/post_processed/err_data/deltat_points.pkl'
if os.path.exists(pklpath1) and os.path.exists(pklpath2) and os.path.exists(pklpath3):
    print 'pickle files exist!'
    fitdata_arr = rw.read_pickle(pklpath1) # (uxtrue, deltat, fitparams, W)... fitparams = [chi, gamma, fit_err[0], fit_err[1], r2]
    ux_points = rw.read_pickle(pklpath2)
    deltat_points = rw.read_pickle(pklpath3)

elif overwrite:
    print 'There is a bug somewhere which induces a segmentation fault: 11. Try saving fit results somewhere, and load them later.'
    rw.write_pickle(pklpath1, fitdata_arr)
    rw.write_pickle(pklpath2, ux_points)
    rw.write_pickle(pklpath3, deltat_points)

    fitdata_arr = rw.read_pickle(pklpath1) # (uxtrue, deltat, fitparams, W)... fitparams = [chi, gamma, alpha, chi_err, gamma_err, alpha_err, r2]
    ux_points = rw.read_pickle(pklpath2)
    deltat_points = rw.read_pickle(pklpath3)

# plt.figure()
# xyz=np.array(np.random.random((100,3)))
# marker_size=15
# plt.scatter(xyz[:,0], xyz[:,1], marker_size, c=xyz[:,2])
# plt.title("Point observations")
# plt.xlabel("Easting")
# plt.ylabel("Northing")
# cbar= plt.colorbar()
# cbar.set_label("elevation (m)", labelpad=+1)
# plt.show()

cmap3 = mpl.cm.get_cmap('viridis')
normalize = mpl.colors.Normalize(vmin=-100, vmax=100)

for iw in range(fitdata_arr.shape[3]):
    fig9, ax9 = plt.subplots(nrows=1, ncols=1)
    for j in range(0, ux_points.shape[0], 10):
        ind = np.abs(fitdata_arr[j, :, 0, iw]) < 300
        ax9.plot(deltat_points[j, ind], fitdata_arr[j, ind, 0, iw], label='$U_{x}^{true} [%.f, %.f)$' % (ux_points[j, 0]-5, ux_points[j, 0]+5), color=cmap3(normalize(ux_points[j, 0])), linewidth=6)
    ax9.set_xlim(0, 0.2)
    ax9.set_ylim(-100, 100)
    ax9.legend(loc=2)
    graph.labelaxes(ax9, '$ \Delta t$ (a.u.)', '$\chi$ (px/unit time)')

    graph.save(dir + '/results/err_dist_nouz/_w%dpx' % iw, ext='png')
    plt.close()
import sys
sys.exit()
#
# setting up fitparams for manipulation

# plotting preparation
grid_deltat, grid_ux = np.mgrid[0.0:0.2:200j, -300:300:1001j]
ind = ~np.isnan(np.ravel(fitdata_arr[..., 0, 0]))
points = np.array(zip(np.ravel(deltat_points)[ind], np.ravel(ux_points)[ind]))

fitparams = np.zeros((fitdata_arr.shape[0], fitdata_arr.shape[1], fitdata_arr.shape[2]))
wind = 0 # index for interrogation window size
print fitdata_arr.shape
for wind in range(fitdata_arr.shape[3]):
    for i in range(fitparams.shape[2]):
        fitparams[..., i] = fitdata_arr[..., i, wind][:]
        fitparams[..., i][fitparams[..., i] > 10**3] = np.nan # replace ridiculous values with nan
        fitparams[..., i][fitparams[..., i] < -10 ** 3] = np.nan

    chi_data = griddata(points, fitparams[..., 0].flatten()[ind], (grid_deltat, grid_ux), method='nearest') # mean of lorentz fit
    gamma_data = griddata(points, fitparams[..., 1].flatten()[ind], (grid_deltat, grid_ux), method='nearest')
    chi_err_data = griddata(points, fitparams[..., 3].flatten()[ind], (grid_deltat, grid_ux), method='nearest')
    gamma_err_data = griddata(points, fitparams[..., 4].flatten()[ind], (grid_deltat, grid_ux), method='nearest')
    r2_data = griddata(points, fitparams[..., 6].flatten()[ind], (grid_deltat, grid_ux), method='nearest')

    # fig8, ax81, cc81 = graph.color_plot(grid_deltat, grid_ux, chi_data, cmap=cmap2, aspect=None, fignum=8, subplot=231, vmin=-300, vmax=300)
    # graph.add_colorbar(cc81, label=r'$\chi$ (px/unit time)', aspect=None)
    #
    # fig8, ax82, cc82 = graph.color_plot(grid_deltat, grid_ux, gamma_data, cmap=cmap2, aspect=None, fignum=8, subplot=232, vmin=-300, vmax=300)
    # graph.add_colorbar(cc82, label=r'$\gamma$ (px/unit time)', aspect=None)
    #
    # fig8, ax83, cc83 = graph.color_plot(grid_deltat, grid_ux, chi_err_data, cmap=cmap2, aspect=None, fignum=8, subplot=233, vmin=-300, vmax=300)
    # graph.add_colorbar(cc83, label=r'$\Delta \chi$ (px/unit time)', aspect=None)
    #
    # fig8, ax84, cc84 = graph.color_plot(grid_deltat, grid_ux, gamma_err_data, cmap=cmap2, aspect=None, fignum=8, subplot=234, vmin=-300, vmax=300)
    # graph.add_colorbar(cc84, label=r'$\Delta \gamma$ (px/unit time)', aspect=None)
    #
    # fig8, ax85, cc85 = graph.color_plot(grid_deltat, grid_ux, r2_data, cmap=cmap, aspect=None, fignum=8, subplot=235, figsize=(25, 10), vmin=0, vmax=1)
    # graph.add_colorbar(cc85, label=r'$R^2$', aspect=None)
    #
    # axes = [ax81, ax82, ax83, ax84, ax85]
    # for ax in axes:
    #     graph.labelaxes(ax, r'$\Delta t$ (a.u.)', r'$U_x^{true}$ (px/unit time)')
    #     ax.set_facecolor('k')
    # fig8.tight_layout()


    fig8, ax81, cc81 = graph.color_plot(grid_deltat, grid_ux, chi_data, cmap=cmap2, aspect=None, fignum=8, subplot=121, vmin=-100, vmax=100)
    # graph.add_colorbar(cc81, label=r'$\chi / |U_x^{true}|$', aspect=None)
    graph.add_colorbar(cc81, label=r'$\chi$ (px/unit time)', aspect=None)

    fig8, ax82, cc82 = graph.color_plot(grid_deltat, grid_ux, gamma_data, cmap='plasma', aspect=None, fignum=8, subplot=122, vmin=0.0, vmax=150, figsize=(20, 10))
    # graph.add_colorbar(cc82, label=r'$\gamma / |U_x^{true}|$', aspect=None)
    graph.add_colorbar(cc82, label=r'$\gamma$ (px/unit time)', aspect=None)

    axes = [ax81, ax82]
    for ax in axes:
        graph.labelaxes(ax, r'$\Delta t$ (a.u.)', r'$U_x^{true}$ (px/unit time)')
        ax.set_facecolor('k')

        ax_2 = ax.twiny()
        ax.set_xlim(0, 0.20)
        ax_2.set_xlabel(r'Normalized time $ \Delta t / \tau$')
        new_tick_locations = np.array([0, tau/4., tau/2.,  tau/4.*3, tau])

        def tick_function(x):
            new_x = x / tau
            return ["%.3f" % z for z in new_x]

        ax_2.set_xlim(ax.get_xlim())
        ax_2.set_xticks(new_tick_locations)
        ax_2.set_xticklabels(tick_function(new_tick_locations))

    iws = [8, 16, 32, 64]
    iw = iws[wind]

    graph.suptitle(r'$W=%d$ px' % iw)
    fig8.tight_layout()

    imgpath8 = dir + '/results/err_dist_heatmap_2/W%dpx' % iw
    # graph.save(imgpath8, 'pdf')
    graph.save(imgpath8, 'png')
    plt.close(fig8)

    # print grid_deltat.shape, grid_ux[:, 0]
    # fig9, ax9 = plt.subplots(nrows=1, ncols=1)
    # for j in range(0, grid_ux.shape[1], 100):
    #     print grid_deltat[:, j], chi_data[:, j]
    #     ax9.scatter(grid_deltat[:, j], chi_data[:, j], label='$U_{x}^{true}=%.f$' % grid_ux[0, j])
    # ax9.legend()
    # plt.show()

    # fig9, ax9 = plt.subplots()
    # chi_data = griddata(points, fitparams[..., 0].flatten()[ind], (grid_deltat, grid_ux),
    #                     method='linear')  # mean of lorentz fit
    # cs = ax9.contour(grid_deltat, grid_ux, chi_data, [-300, -200, -100, -50, 50, 100, 200, 300])
    # ax9.clabel(cs, inline=1, fontsize=10)
    # plt.show()


