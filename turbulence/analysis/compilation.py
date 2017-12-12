#### Compile the measurements for a given set of data
#

import turbulence.display.graphes as graphes
import numpy as np

import turbulence.analysis.Fourier as Fourier
import turbulence.tools.Smath as Smath

import matplotlib


def vortex_collider(M, indices=range(20, 430), version=0):
    # additionnal parameters :
    M.add_param('freq', 'Hz')
    M.add_param('v', 'mms')

    # suffix = graphes.set_name(M,param=['freq','v'])
    # savedir = '/Users/stephane/Documents/Experiences_local/Results/Vortex_collision/'+M.Id.date+'/'+M.Id.get_id()+'/'
    # figs = {}

    # compute additionnal fields
    M.get('enstrophy')
    M.get('dU')
    M.get('omega')
    M.get('E')

    make_figures(M, indices=indices, version=version)


def comparison(Mlist, indices=None, version=0, save=False):
    M = Mlist[0]
    #    suffix = graphes.set_name(M,param=['freq','v'])
    savedir = '/Users/stephane/Documents/Experiences_local/Results/Vortex_collision/' + M.Id.date + '/'
    subdir = 'Summary/'

    ######### Comparison spectrum ##############
    for M, ind in zip(Mlist, indices):
        figs = spectrum_1d(M, indices=ind, norm_factor=(M.param.v / 100.) ** (2. / 3))
    graphes.save_figs(figs, savedir=savedir + subdir, prefix='norm', suffix=str(version))
    # graphes.plt.close('all')


def make_figures(M, indices=None, version=0):
    suffix = graphes.set_name(M, param=['freq', 'v'])
    savedir = '/Users/stephane/Documents/Experiences_local/Results/Vortex_collision/' + M.Id.date + '/' + M.Id.get_id() + '/'

    ###### Field examples ##############
    subdir = 'Example/'
    figs = example(M, i=None)
    graphes.save_figs(figs, savedir=savedir + subdir, prefix=suffix)
    graphes.plt.close('all')
    ####### Time average fields #############
    subdir = 'Time average/'
    figs = time_average(M, indices=indices)
    graphes.save_figs(figs, savedir=savedir + subdir, prefix=suffix)
    graphes.plt.close('all')

    ######## Energy vs time ##############
    subdir = 'Spatial average/'
    figs = spatial_average(M, indices=indices)
    graphes.save_figs(figs, savedir=savedir + subdir, prefix=suffix)
    graphes.plt.close('all')

    ######### 2d spectrum, average in time ##############
    subdir = 'Spectrum/'
    figs = spectrum_2d(M, indices=indices)
    graphes.save_figs(figs, savedir=savedir + subdir, prefix=suffix)
    graphes.plt.close('all')

    ######### 1d spectrum, average in time ##############
    subdir = 'Spectrum/'
    figs = spectrum_1d(M, indices=indices)
    graphes.save_figs(figs, savedir=savedir + subdir, prefix=suffix, suffix=str(version))
    graphes.plt.close('all')


def example(M, i=None):
    figs = {}
    if i == None:
        nx, ny, nt = M.shape()
        i = nt // 4

    fields, names, vmin, vmax, labels, units = std_fields()
    for j, field in enumerate(fields):
        figs.update(graphes.Mplot(M, field, i, fignum=j + 1, colorbar=True, vmin=vmin[j], vmax=vmax[j]))
        ### add label on colorbar
    return figs


def time_average(M, indices=None):
    figs = {}
    fields, names, vmin, vmax, labels, units = std_fields()

    for j, field in enumerate(fields):
        Y = np.nanmean(getattr(M, field)[..., indices], axis=2)
        graphes.color_plot(M.x, M.y, Y, fignum=j + 1, vmin=vmin[j] / 5, vmax=vmax[j] / 5)
        graphes.colorbar(label=names[j] + ' ' + units[j])
        figs.update(graphes.legende('X (mm)', 'Y (mm)', 'Time averaged ' + field, cplot=True))
    return figs


def spatial_average(M, indices=None):
    figs = {}
    fields, names, vmin, vmax, labels, units = std_fields()
    for j, field in enumerate(fields):
        Y_moy = np.nanmean(getattr(M, field), axis=(0, 1))
        graphes.graph(M.t, Y_moy, label=labels[j], fignum=j + 1)
        # graphes.set_axis(0,5,0,18000)
        figs.update(graphes.legende('Time (s)', names[j] + ' (' + units[j] + ')', ''))
    return figs


def spectrum_2d(M, indices=None):
    Fourier.compute_spectrum_2d(M, Dt=3)  # smooth on 3 time step.
    S_E = np.nanmean(M.S_E[..., indices], axis=2)
    graphes.color_plot(M.kx, M.ky, S_E, log=True, fignum=1)
    graphes.colorbar(label='$E_k$')
    figs = graphes.legende('$k_x$ (mm)', '$k_y$ (mm)', 'Energy Spectrum (log)')
    return figs


def spectrum_1d(M, indices=None, norm_factor=1):
    Fourier.compute_spectrum_1d(M, Dt=3)
    S_k = np.nanmean(M.S_k[..., indices], axis=1) / norm_factor
    graphes.graphloglog(M.k, S_k, label='^-', fignum=1)

    k0 = 0.1
    i = np.argmin(np.abs(M.k - k0))
    A = S_k[i] * k0 ** (5. / 3)

    #    print('Total energy : '+np.sum(M.k*S_k))

    graphes.graph(M.k, A * M.k ** (-5. / 3), label='r--')
    figs = graphes.legende('$k$ (mm$^{-1}$)', 'E (m/s^2)', '')
    return figs


def spatial_decay(M, field='E', indices=None):
    from mpl_toolkits.axes_grid.inset_locator import inset_axes
    import stephane.vortex.track as track

    fields, names, vmin, vmax, labels, units = std_fields()
    j = fields.index(field)

    Z = np.nanmean(getattr(M, field)[..., indices], axis=2)

    X, Y = [], []
    for i in indices:
        tup = track.positions(M, i, field=field, indices=indices, step=1, sigma=10.)
        X.append(tup[1])
        Y.append(tup[3])
    X0, Y0 = np.nanmean(X), np.nanmean(Y)

    R, Theta = Smath.cart2pol(M.x - X0, M.y - Y0)
    R0 = M.param.Diameter

    Z_flat = np.ndarray.flatten(Z)
    R_flat = np.ndarray.flatten(R)
    Theta_flat = np.ndarray.flatten(Theta)

    phi = np.pi / 2
    C = np.mod((Theta_flat + phi + np.pi) / 2 / np.pi, 1)
    cmap = matplotlib.cm.hot
    color = [matplotlib.colors.rgb2hex(cmap(c)[:3]) for c in C]

    fig, ax2 = graphes.set_fig(1, subplot=122)
    fig.set_size_inches(20, 6)
    ax2.scatter(R_flat, Z_flat, marker='o', facecolor=color, alpha=0.3, lw=0, cmap=cmap)

    Rth = np.arange(10 ** 1, 10 ** 2, 1.)
    graphes.graphloglog(Rth, 10 ** 5 * (Rth / R0) ** -3.2, label='k--')
    graphes.graphloglog(Rth, 5 * 10 ** 4 * (Rth / R0) ** -4.5, label='k--')
    graphes.set_axis(10 ** 0, 10 ** 2, 10 ** 2, 8 * 10 ** 4)
    figs = graphes.legende('$R$ (mm)', 'Energy (mm$^2$/s$^{2}$)', 'Spatial decay')

    fig, ax1 = graphes.set_fig(1, subplot=121)
    graphes.color_plot(M.x, M.y, Z, fignum=1, vmin=0, vmax=40000, subplot=121)
    graphes.colorbar(label=names[j] + ' (' + units[j] + ')')
    figs.update(graphes.legende('X (mm)', 'Y (mm)', 'Time averaged ' + field, cplot=True))

    inset_ax = inset_axes(ax2, height="50%", width="50%", loc=3)
    inset_ax.pcolormesh(M.x / 10 - 10, M.y / 10 - 10, Theta, cmap=cmap)
    inset_ax.axis('off')

    return figs


#    graphes.save_figs(figs,savedir=savedir,prefix='Final',suffix = 'Colored_Scaling_Exponent_from_32d_to45d')

def std_fields():
    fields = ['Ux', 'Uy', 'omega', 'E', 'enstrophy']
    names = ['$U_x$', '$U_y$', '$\omega_z$', '$E$', '$\omega^2$']
    vmin = [-300, -300, -300, 0, 0]
    vmax = [300, 300, 300, 10 ** 5, 5 * 10 ** 4]
    labels = ['>-', '^-', 'o-', 's-', 'p-']
    units = ['mm/s', 'mm/s', 's$^{-1}$', 'mm$^2$/s$^2$', 's$^{-2}$']
    return fields, names, vmin, vmax, labels, units
