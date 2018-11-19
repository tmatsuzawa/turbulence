import argparse
import glob
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import library.tools.rw_data as rw
import turbulence.jhtd.get as jhtd_get
import turbulence.jhtd.tools as jhtd_tools
import library.basics.formatarray as fa
import library.display.graph as graph

parser = argparse.ArgumentParser(description='Make a hdf5 file out of PIVLab txt outputs')
parser.add_argument('-datadir', '--datadir', help='Parent directory of PIVlab output HDF5 directories', type=str,
                    default='/Volumes/labshared3-1/takumi/JHTD-sample/JHT_Database/Data/synthetic_data_from_bob/double_oseen_PIV_gamma17000_NP50000_r40_D150_cx640_cy400_fps500_w1280_h800_psize3_dx200/hdf5data')
parser.add_argument('-fdp', '--fakedatapath', help='full path to fake data', type=str,
                    default='/Volumes/labshared3-1/takumi/JHTD-sample/JHT_Database/Data/synthetic_data_from_bob/double_oseen_PIV_gamma17000_NP50000_r40_D150_cx640_cy400_fps500_w1280_h800_psize3_dx200/hdf5data/double_oseen_PIV_gamma17000_NP50000_r40_D150_cx640_cy400_fps500_w1280_h800_psize3_dx200.h5')
parser.add_argument('-imw', '--imw', help='width of fake data image', type=int, default=1024)
parser.add_argument('-imh', '--imh', help='height of fake data image', type=int, default=1024)
parser.add_argument('-iw', '--iw', help='interrogation window width, default:16 (px)', type=int, default=16)

args = parser.parse_args()

# Data architecture
pivdatadir = args.datadir

parentdir = os.path.dirname(pivdatadir)
resultdir = os.path.join(parentdir, 'comp_fakedata_pivlab/')

# fake data image size
imsize = (args.imh, args.imw)
dx, dy, dz = 1., 1., 1.


# Plotting settings
cmap = 'RdBu'
# cmap = 'magma'
params = {'figure.figsize': (18, 14),
          'xtick.labelsize': 14,  # tick
          'ytick.labelsize': 14
          }

graph.update_figure_params(params)



# Load fake data
fkp = args.fakedatapath

data = rw.read_hdf5(fkp)
xx, yy = data['x'], data['y']
ux0, uy0 = data['ux'], data['uy']




# Coarse-grain data
nrows_sub, ncolumns_sub = args.iw, args.iw # number of pixels to average over
xx_coarse = fa.coarse_grain_2darr_overwrap(xx, nrows_sub, ncolumns_sub, overwrap=0.5)
yy_coarse = fa.coarse_grain_2darr_overwrap(yy, nrows_sub, ncolumns_sub, overwrap=0.5)
ux0_coarse = fa.coarse_grain_2darr_overwrap(ux0, nrows_sub, ncolumns_sub, overwrap=0.5)
uy0_coarse = fa.coarse_grain_2darr_overwrap(uy0, nrows_sub, ncolumns_sub, overwrap=0.5)

fig1, ax11, cc11 = graph.color_plot(xx, yy, ux0, cmap=cmap, vmin=-2, vmax=2,  fignum=1, subplot=221)
fig1, ax12, cc12 = graph.color_plot(xx_coarse, yy_coarse, ux0_coarse, cmap=cmap, vmin=-2, vmax=2,  fignum=1, subplot=222)
fig1, ax13, cc13 = graph.color_plot(xx, yy, uy0, cmap=cmap, vmin=-2, vmax=2,  fignum=1, subplot=223)
fig1, ax14, cc14 = graph.color_plot(xx_coarse, yy_coarse, uy0_coarse, cmap=cmap, vmin=-2, vmax=2,  fignum=1, subplot=224, figsize=(18, 14))
axes1 = [ax11, ax12, ax13, ax14]
ccs1 = [cc11, cc12, cc13, cc14]
titles1 = ['Original $U_x$', 'Coarse-grained $U_x$', 'Original $U_y$', 'Coarse-grained $U_y$']
for ax, cc, title in zip(axes1, ccs1, titles1):
    graph.add_colorbar(cc, ax=ax, ticklabelsize=10)
    graph.title(ax, title)
    # graph.setaxes(ax, 0, 2*np.pi, 0, 2*np.pi)
    graph.labelaxes(ax, '$X$ (a.u.)', '$Y$ (a.u.)')
    if cmap == 'RdBu':
        ax.set_facecolor('k')
graph.suptitle('Fake data')
filename = 'fake_data_vel_fields_%s' %cmap
graph.save(resultdir + filename)
plt.close()


################
# PIV-processed data
################
# data architecture
pivlab_outputs = glob.glob(pivdatadir + '/PIV*')
pivlab_output_names = [os.path.split(filename)[1] for filename in pivlab_outputs]
titles2 = ['No. of particles: %s' % filename[-8:-3] for filename in pivlab_output_names]
axes2, ccs2 = [], []
# fig = plt.figure(num=2, figsize=(18, 18))
for i, pivlab_output in enumerate(pivlab_outputs):
    print pivlab_output
    data = rw.read_hdf5(pivlab_output)
    xx, yy = np.array(data['x']) * dx, np.array(data['y']) * dy
    ux, uy = np.array(data['ux'])[..., 0], np.array(data['uy'])[..., 0]
    label = '# of particles: %s' % pivlab_output[-8:-3]
    fig2, ax2, cc2 = graph.color_plot(xx, yy, ux, cmap=cmap, vmin=-2, vmax=2, fignum=2,  subplot=(221+i), figsize=(18, 14) )
    # ax2.quiver(xx, yy, ux, uy)
    axes2.append(ax2)
    ccs2.append(cc2)

    # get piv data size
    pivdatasize = ux.shape
    data.close()

for ax, cc, title in zip(axes2, ccs2, titles2):
    graph.add_colorbar(cc, ax=ax, ticklabelsize=10)
    graph.title(ax, title)
    #graph.setaxes(ax, 0, 2*np.pi, 0, 2*np.pi)
    graph.labelaxes(ax, '$X$ (a.u.)', '$Y$ (a.u.)')
    if cmap == 'RdBu':
        ax.set_facecolor('k')
graph.suptitle('PIVLab $U_x$ (W=16, pre- & post-processed)')

filename = 'pivlab_ux_%s' %cmap
graph.save(resultdir + filename)

# Fig 3: Uy
titles3 = ['No. of particles: %s' % filename[-8:-3] for filename in pivlab_output_names]
axes3, ccs3 = [], []
for i, pivlab_output in enumerate(pivlab_outputs):
    data = rw.read_hdf5(pivlab_output)
    xx, yy = np.array(data['x']) * dx, np.array(data['y']) * dy
    ux, uy = np.array(data['ux'])[..., 0], np.array(data['uy'])[..., 0]
    label = '# of particles: %s' % pivlab_output[-8:-3]
    fig3, ax3, cc3 = graph.color_plot(xx, yy, uy, cmap=cmap, vmin=-2, vmax=2, fignum=3,  subplot=(221+i), figsize=(18, 14) )
    # ax2.quiver(xx, yy, ux, uy)
    axes3.append(ax3)
    ccs3.append(cc3)
    data.close()

for ax, cc, title in zip(axes3, ccs3, titles3):
    graph.add_colorbar(cc, ax=ax, ticklabelsize=10)
    graph.title(ax, title)
    #graph.setaxes(ax, 0, 2*np.pi, 0, 2*np.pi)
    graph.labelaxes(ax, '$X$ (a.u.)', '$Y$ (a.u.)')
    if cmap == 'RdBu':
        ax.set_facecolor('k')
graph.suptitle('PIVLab $U_y$ (W=16, pre- & post-processed)')

filename = 'pivlab_uy_%s' %cmap
graph.save(resultdir + filename)


##########
# Difference
# Fig 4: Ux (% diff), Fig 5: Uy (% diff)
# Fig 13: Ux (diff), Fig 14: Uy (diff)
##########
titles4 = ['No. of particles: %s' % filename[-8:-3] for filename in pivlab_output_names]
titles5 = ['No. of particles: %s' % filename[-8:-3] for filename in pivlab_output_names]
titles13 = ['No. of particles: %s' % filename[-8:-3] for filename in pivlab_output_names]
titles14 = ['No. of particles: %s' % filename[-8:-3] for filename in pivlab_output_names]
axes4, ccs4, axes5, ccs5 = [], [], [], []
axes13, ccs13, axes14, ccs14 = [], [], [], []
for i, pivlab_output in enumerate(pivlab_outputs):
    data = rw.read_hdf5(pivlab_output)
    xx, yy = np.array(data['x']) * dx, np.array(data['y']) * dy
    ux, uy = np.array(data['ux'])[..., 0], np.array(data['uy'])[..., 0]
    label = '# of particles: %s' % pivlab_output[-8:-3]
    fig4, ax4, cc4 = graph.color_plot(xx, yy, (ux-ux0_coarse) / ux0_coarse*100, cmap=cmap, vmin=-200, vmax=200, fignum=4, subplot=(221 + i), figsize=(18, 14))
    fig5, ax5, cc5 = graph.color_plot(xx, yy, (uy-uy0_coarse) / uy0_coarse * 100, cmap=cmap, vmin=-200, vmax=200, fignum=5, subplot=(221 + i), figsize=(18, 14))
    fig13, ax13, cc13 = graph.color_plot(xx, yy, (ux-ux0_coarse), cmap=cmap, vmin=-2, vmax=2, fignum=13, subplot=(221 + i), figsize=(18, 14))
    fig14, ax14, cc14 = graph.color_plot(xx, yy, (uy-uy0_coarse), cmap=cmap, vmin=-2, vmax=2, fignum=14, subplot=(221 + i), figsize=(18, 14))
    axes4.append(ax4)
    ccs4.append(cc4)
    axes5.append(ax5)
    ccs5.append(cc5)
    axes13.append(ax13)
    ccs13.append(cc13)
    axes14.append(ax14)
    ccs14.append(cc14)
    data.close()

plt.figure(4)
for ax, cc, title in zip(axes4, ccs4, titles4):
    graph.add_colorbar(cc, ax=ax, ticklabelsize=10)
    graph.title(ax, title)
    #graph.setaxes(ax, 0, 2*np.pi, 0, 2*np.pi)
    graph.labelaxes(ax, '$X$ (a.u.)', '$Y$ (a.u.)')
    if cmap == 'RdBu':
        ax.set_facecolor('k')
graph.suptitle('% difference $U_x$ (W=16, pre- & post-processed)', fignum=4)

plt.figure(5)
for ax, cc, title in zip(axes5, ccs5, titles5):
    graph.add_colorbar(cc, ax=ax, ticklabelsize=10)
    graph.title(ax, title)
    #graph.setaxes(ax, 0, 2*np.pi, 0, 2*np.pi)
    graph.labelaxes(ax, '$X$ (a.u.)', '$Y$ (a.u.)')
    if cmap == 'RdBu':
        ax.set_facecolor('k')
graph.suptitle('% difference $U_y$ (W=16, pre- & post-processed)')

plt.figure(13)
for ax, cc, title in zip(axes13, ccs13, titles13):
    graph.add_colorbar(cc, ax=ax, ticklabelsize=10)
    graph.title(ax, title)
    #graph.setaxes(ax, 0, 2*np.pi, 0, 2*np.pi)
    graph.labelaxes(ax, '$X$ (a.u.)', '$Y$ (a.u.)')
    if cmap == 'RdBu':
        ax.set_facecolor('k')
graph.suptitle('Difference $U_x$ (W=16, pre- & post-processed)')

plt.figure(14)
for ax, cc, title in zip(axes14, ccs14, titles14):
    graph.add_colorbar(cc, ax=ax, ticklabelsize=10)
    graph.title(ax, title)
    #graph.setaxes(ax, 0, 2*np.pi, 0, 2*np.pi)
    graph.labelaxes(ax, '$X$ (a.u.)', '$Y$ (a.u.)')
    if cmap == 'RdBu':
        ax.set_facecolor('k')
graph.suptitle('Difference $U_y$ (W=16, pre- & post-processed)')

filename = 'percent_diff_ux_%s' % cmap
graph.save(resultdir + filename, fignum=4)

filename = 'percent_diff_uy_%s' % cmap
graph.save(resultdir + filename, fignum=5)

filename = 'diff_ux_%s' % cmap
graph.save(resultdir + filename, fignum=13)

filename = 'diff_uy_%s' % cmap
graph.save(resultdir + filename, fignum=14)




##########
# PDF comparison (mean, variance, skewness, flatness)
# Fig 6: Ux, Fig 7: Uy
nnon_nans_ux_list, n_particles_list = [], []
fig6, ax6 = graph.pdf(ux0, nbins=200, fignum=6, label='Fake data')
fig7, ax7 = graph.pdf(uy0, nbins=200, fignum=7, label='Fake data')
linestyles = ['--', '--', '--', '--', '--', '--', '--', '--']
for i, pivlab_output in enumerate(pivlab_outputs):
    data = rw.read_hdf5(pivlab_output)
    xx, yy = np.array(data['x']) * dx, np.array(data['y']) * dy
    ux, uy = np.array(data['ux'])[..., 0], np.array(data['uy'])[..., 0]
    label = 'N: %s' % pivlab_output[-8:-3]
    nnon_nans_ux = np.count_nonzero(~np.isnan(ux))
    nnon_nans_ux_list.append(nnon_nans_ux)
    n_particles_list.append(float(pivlab_output[-8:-3]))
    # get piv data size
    pivdatasize = ux.shape
    print nnon_nans_ux
    if nnon_nans_ux / 4. < 200:
        nbins = int(nnon_nans_ux * 0.25)
    else:
        nbins = 200
    fig6, ax6 = graph.pdf(ux, nbins=nbins, fignum=6, label=label, alpha=0.7, linestyle=linestyles[i])
    fig7, ax7 = graph.pdf(uy, nbins=nbins, fignum=7, label=label, alpha=0.7, linestyle=linestyles[i])
    data.close()

plt.figure(6)
ax6.legend()
graph.labelaxes(ax6, '$U_x$ (a.u.)', 'Prob. density')
graph.setaxes(ax6, -4, 4, -0.1, 2)
filename = 'pdf_ux'
graph.save(resultdir + filename)

plt.figure(7)
ax7.legend()
graph.labelaxes(ax7, '$U_y$ (a.u.)', 'Prob. density')
graph.setaxes(ax7, -4, 4, -0.1, 2)
filename = 'pdf_uy'
graph.save(resultdir + filename)




# Number of nans as a function of density/number of particles
# Fig 8: Number of nans
plt.figure(num=8, figsize=(8, 8))
rhos = np.array(n_particles_list) / float((imsize[0] * imsize[1]))
fig8, ax8 = graph.scatter(rhos, 1 - (np.array(nnon_nans_ux_list) / float((pivdatasize[0]*pivdatasize[1]))), fignum=8)
graph.tosemilogx(ax8)
graph.setaxes(ax8, 7 * 10**(-5), 10**-1, 0, 1)

ax8_2 = ax8.twiny()

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        if int(base) == 1:
            return r"$10^{{{0}}}$".format(int(exponent))
        else:
            return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))

    else:
        return float_str

def tick_function(X):
    n_tps = np.array(X) * float((imsize[0] * imsize[1]))
    return [latex_float(n_tp) for n_tp in n_tps]

# dummy plot
ax8_2.scatter(rhos, 1 - (np.array(nnon_nans_ux_list) / float((pivdatasize[0]*pivdatasize[1]))), s=0)
graph.tosemilogx(ax8_2)
ax8_2.set_xlim(ax6.get_xlim())
new_tick_locations = [rhos[0], rhos[1], rhos[3], 10**5 / float((imsize[0] * imsize[1]))]
ax8_2.set_xticks(new_tick_locations)
ax8_2.set_xticklabels(tick_function(new_tick_locations))
ax8_2.set_xlabel(r"No. of tracer particles in (%d px, %d px)" % (imsize[0], imsize[1]))

graph.labelaxes(ax8, 'Tracer particle density (px$^{-2}$)', 'Fraction of nans')

filename = 'frac_nans_ux'
graph.save(resultdir + filename)



# Noise PDF
# Fig 9: Scaled Noise PDF varying particle density (DeltaUx / Ux)
# Fig 12: Scaled Noise PDF varying particle density (DeltaUy / Uy)
# Fig 15: Gamma (width of Lorentzian fit) as a function of particle number
# Fig 16: Noise PDF varying particle density (DeltaUx)
# Fig 17: Noise PDF varying particle density (DeltaUy)

import library.basics.std_func as std_func
titles9 = ['No. of particles: %s' % filename[-8:-3] for filename in pivlab_output_names]
gammas_ux, ux0s, ux0_fit_errs, gammas_uy, uy0s, uy0_fit_errs, ntps = [], [], [], [], [], [], []
for i, pivlab_output in enumerate(pivlab_outputs):
    data = rw.read_hdf5(pivlab_output)
    xx, yy = np.array(data['x']) * dx, np.array(data['y']) * dy
    ux, uy = np.array(data['ux'])[..., 0], np.array(data['uy'])[..., 0]
    noise_ux = (ux - ux0_coarse) / ux0_coarse
    noise_uy = (uy - uy0_coarse) / uy0_coarse
    label = 'N: %s' % pivlab_output[-8:-3]
    ntps.append(float(pivlab_output[-8:-3]))
    nnon_nans_ux = np.count_nonzero(~np.isnan(ux))
    # get piv data size
    pivdatasize = ux.shape

    if nnon_nans_ux / 4. < 200:
        nbins = int(nnon_nans_ux * 0.25)
        nbins_ux, nbins_uy = nbins, nbins
    else:
        nbins = nnon_nans_ux / 4
        noise_ux_min, noise_uy_min = np.nanmin(noise_ux), np.nanmin(noise_uy)
        noise_ux_max, noise_uy_max = np.nanmax(noise_ux), np.nanmax(noise_uy)
        nbins_ux = int((noise_ux_max - noise_ux_min) / 0.05)
        nbins_uy = int((noise_uy_max - noise_uy_min) / 0.05)

    fig9, ax9, bins, hist = graph.pdf(noise_ux, nbins=nbins_ux, label=label, alpha=0.6,  fignum=9, return_data=True)
    indices = np.where((bins > -4) & (bins < 4))
    popt, pcov = curve_fit(std_func.lorentzian_norm, bins[indices], hist[indices])

    ux0s.append(popt[0])
    gammas_ux.append(popt[1])
    ux0_fit_errs.append((np.sqrt(np.diag(pcov)))[0])
    fig12, ax12, bins, hist = graph.pdf(noise_uy, nbins=nbins_uy, label=label, alpha=0.6,  fignum=12, return_data=True)
    indices = np.where((bins > -4) & (bins < 4))
    popt, pcov = curve_fit(std_func.lorentzian_norm, bins[indices], hist[indices])
    uy0s.append(popt[0])
    gammas_uy.append(popt[1])
    uy0_fit_errs.append((np.sqrt(np.diag(pcov))))

    fig16, ax16, bins, hist = graph.pdf(ux - ux0_coarse, nbins=200, label=label, alpha=0.6, fignum=16, return_data=True)
    fig17, ax17, bins, hist = graph.pdf(uy - uy0_coarse, nbins=200, label=label, alpha=0.6, fignum=17, return_data=True)


    data.close()

plt.figure(9)
ax9.legend(loc=1)
graph.setaxes(ax9, -4, 4, -0.1, 2)
graph.labelaxes(ax9, r'$\Delta U_x(\vec{r}) / U_x(\vec{r}) $', 'Prob. density')
filename = 'pivlab_scaled_noise_ux_varying_density'
graph.save(resultdir + filename)

plt.figure(12)
ax12.legend(loc=1)
ax12.set_xlim(-4, 4)
graph.setaxes(ax12, -4, 4, -0.1, 2)
graph.labelaxes(ax12, r'$\Delta U_y(\vec{r}) / U_y(\vec{r}) $', 'Prob. density')
filename = 'pivlab_scaled_noise_uy_varying_density'
graph.save(resultdir + filename)


# Width of Lorentzian curve as a function of number of tracer particles
fig15, ax15, _ = graph.errorfill(ntps, gammas_ux, ux0_fit_errs[1], label='$U_x$',fignum=15)
fig15, ax15, _ = graph.errorfill(ntps, gammas_uy, uy0_fit_errs[1], label='$U_y$', fignum=15)
graph.labelaxes(ax15, r'No. of tracer particles in (%d px, %d px)' % (imsize[0], imsize[1]), '$\gamma$ (a.u.)')
graph.setaxes(ax15, 0, 11000, 0, 0.5)
ax15.legend(loc=1, fontsize=18)
filename = 'lorentzian_fit_gammas_scaled_noise'
graph.save(resultdir + filename)

plt.figure(16)
ax16.legend(loc=1)
graph.setaxes(ax16, -4, 4, -0.1, 3.5)
graph.labelaxes(ax16, r'$\Delta U_x $ (a.u.)', 'Prob. density')
filename = 'pivlab_noise_ux_varying_density'
graph.save(resultdir + filename)

plt.figure(17)
ax17.legend(loc=1)
graph.setaxes(ax17, -4, 4, -0.1, 3.5)
graph.labelaxes(ax17, r'$\Delta U_y $ (a.u.)', 'Prob. density')
filename = 'pivlab_noise_uy_varying_density'
graph.save(resultdir + filename)
#
# #
#
#

## Lorentzian Fit example
## Lorentzian fit of noise PDF
noise_ux = (ux-ux0_coarse)/ux0_coarse
noise_uy = (uy-uy0_coarse) / uy0_coarse
fig10, ax10, bins, hist = graph.pdf(noise_ux, nbins=nbins_ux, label='$U_x$, N=10000', fignum=10, return_data=True)
# Use bins where -4 < bins < bins for curve fit
indices = np.where((bins > -4) & (bins < 4))
fig10, ax10, popt, pcov = graph.plot_fit_curve(bins[indices], hist[indices], func=std_func.lorentzian_norm, label=r'Lorentzian fit', fignum=10)
# fig10, ax10, popt, pcov = graph.plot_fit_curve(bins, hist, func=std_func.gaussian_norm, label=r'Gaussian fit $U_x$', fignum=10)
ax10.legend(loc=1)
graph.setaxes(ax10, -4, 4, -0.1, 2)
graph.labelaxes(ax10, r'$\Delta U_x(\vec{r}) / U_x(\vec{r}) $', 'Prob. density')
text = r'$(x_0, \gamma)=(%.2f, %.2f)$' % (popt[0], popt[1])
graph.addtext(ax10, text=text, option='tl', color='C1')
filename = 'pivlab_noise_ux'
graph.save(resultdir + filename)

fig11, ax11, bins, hist = graph.pdf(noise_uy, nbins=nbins_uy, label='$U_y$, N=10000', fignum=11, return_data=True)
# Use bins where -4 < bins < bins for curve fit
indices = np.where((bins > -4) & (bins < 4))
fig11, ax11, popt, pcov = graph.plot_fit_curve(bins[indices], hist[indices], func=std_func.lorentzian_norm, label=r'Lorentzian fit', fignum=11)
# fig11, ax11, popt, pcov = graph.plot_fit_curve(bins, hist, func=std_func.gaussian_norm, label=r'Gaussian fit $U_y$', fignum=11)
ax11.legend(loc=1)
graph.setaxes(ax11, -4, 4, -0.1, 2)
graph.labelaxes(ax11, r'$\Delta U_y(\vec{r}) / U_y(\vec{r}) $', 'Prob. density')
text = r'$(x_0, \gamma)=(%.2f, %.2f)$' % (popt[0], popt[1])
graph.addtext(ax11, text=text, option='tl', color='C1')
filename = 'pivlab_noise_uy'
graph.save(resultdir + filename)





## Covariance (Signal-Noise)
# Fig 18: Noise-Signal covariance
import library.tools.process_data as process
titles18 = ['No. of particles: %s' % filename[-8:-3] for filename in pivlab_output_names]
for i, pivlab_output in enumerate(pivlab_outputs):
    data = rw.read_hdf5(pivlab_output)
    xx, yy = np.array(data['x']) * dx, np.array(data['y']) * dy
    ux, uy = np.array(data['ux'])[..., 0], np.array(data['uy'])[..., 0]
    label = 'N: %s' % pivlab_output[-8:-3]

    fig18, ax18 = graph.scatter(ux.flatten(), (ux - ux0_coarse).flatten(), alpha=0.2, fignum=18, subplot=121, figsize=(16, 8))
    fig18, ax19 = graph.scatter(uy.flatten(), (uy - uy0_coarse).flatten(), color='C1', alpha=0.2, fignum=18, subplot=122, figsize=(16, 8))
    mask1, mask2 = process.get_mask_for_nan_and_inf(ux), process.get_mask_for_nan_and_inf((ux - ux0_coarse))
    mask3, mask4 = process.get_mask_for_nan_and_inf(ux), process.get_mask_for_nan_and_inf((ux - ux0_coarse))
    ux_clean, ux_err_clean = process.delete_masked_elements(ux, mask1), process.delete_masked_elements(
        (ux - ux0_coarse), mask2)
    uy_clean, uy_err_clean = process.delete_masked_elements(uy, mask3), process.delete_masked_elements(
        (uy - uy0_coarse), mask4)
    cov_ux = np.corrcoef([ux_clean, ux_err_clean])  # covariance matrix
    cov_uy = np.corrcoef([uy_clean, uy_err_clean])
    text1 = 'Corr. Coeff.: %.3f' % cov_ux[1, 0]
    text2 = 'Corr. Coeff.: %.3f' % cov_uy[1, 0]
    graph.setaxes(ax18, -15, 15, -15, 15)
    graph.setaxes(ax19, -15, 15, -15, 15)
    graph.addtext(ax18, text=text1, option='tl', color='C0')
    graph.addtext(ax19, text=text2, option='tl', color='C1')
    graph.labelaxes(ax18, r'$U_x$ (a.u.)', r'$\Delta U_x$ (a.u.)')
    graph.labelaxes(ax19, r'$U_y$ (a.u.)', r'$\Delta U_y$ (a.u.)')
    graph.suptitle(titles18[i])

    filename = 'pivlab_noise_signal_correlation_N%s' % pivlab_output[-8:-3]
    graph.save(resultdir + filename)
    plt.close(fig18)

    data.close()



fig18, ax18 = graph.scatter(ux.flatten(), (ux-ux0_coarse).flatten(), alpha=0.2, fignum=18, subplot=121, figsize=(16, 8))
fig18, ax19 = graph.scatter(uy.flatten(), (uy-uy0_coarse).flatten(), color='C1', alpha=0.2, fignum=18, subplot=122, figsize=(16, 8))

mask1, mask2 = process.get_mask_for_nan_and_inf(ux), process.get_mask_for_nan_and_inf((ux-ux0_coarse))
mask3, mask4 = process.get_mask_for_nan_and_inf(ux), process.get_mask_for_nan_and_inf((ux-ux0_coarse))
ux_clean, ux_err_clean = process.delete_masked_elements(ux, mask1), process.delete_masked_elements((ux-ux0_coarse), mask2)
uy_clean, uy_err_clean = process.delete_masked_elements(uy, mask3), process.delete_masked_elements((uy-uy0_coarse), mask4)
cov_ux = np.corrcoef([ux_clean, ux_err_clean]) # covariance matrix
cov_uy = np.corrcoef([uy_clean, uy_err_clean])
text1 = 'Corr. Coeff.: %.3f' % cov_ux[1, 0]
text2 = 'Corr. Coeff.: %.3f' % cov_uy[1, 0]
graph.setaxes(ax18, -15, 15, -15, 15)
graph.setaxes(ax19, -15, 15, -15, 15)
graph.addtext(ax18, text=text1, option='tl', color='C0')
graph.addtext(ax19, text=text2, option='tl', color='C1')
graph.labelaxes(ax18, r'$U_x$ (a.u.)', r'$\Delta U_x$ (a.u.)')
graph.labelaxes(ax19, r'$U_y$ (a.u.)', r'$\Delta U_y$ (a.u.)')
fig18.tight_layout()
filename = 'pivlab_noise_signal_correlation'
graph.save(resultdir + filename)


graph.show()