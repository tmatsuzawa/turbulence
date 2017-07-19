import turbulence.mdata.M_manip as M_manip
import turbulence.tools.rw_data as rw_data
import glob
import turbulence.tools.browse as browse
import turbulence.pprocess.check_piv as check
import turbulence.mdata.Mdata_PIVlab as pivlab
import turbulence.mdata.Sdata_manip as Sdata_manip
import turbulence.mdata.Sdata as Sdata
import turbulence.analysis.cdata as cdata
import numpy as np
import turbulence.pprocess.test_serie as tests
import turbulence.display.graphes as graphes
import turbulence.display.panel as panel
import turbulence.vortex.track as track
import turbulence.manager.access as access
import turbulence.analysis.Fourier as Fourier
import turbulence.analysis.compilation as comp
import turbulence.tools.Smath as Smath
import matplotlib.pyplot as plt
import sys

'''Examine (time averaged) properties of experiment data
Eventually we will roll these into methods of a vortex_collision class, but for now it is a script
'''

date = '2017_06_30'
# savedir = '/Users/stephane/Documents/Experiences_local/Results/Vortex_collision/'+date+'/'
savedir = '/Users/npmitchell/Desktop/data_local/vortex_collision/' + date + '/'
subdir = 'summary/'

indices = [2]  # range(6,10)
# load the Sdata from 0 to 5. If you want other datam just change the range.
# Check with the file name by typing Slist[0].fileCine

# Slist is Sdata_date_index_ver.hdf5.
# Each index of Slist is associated to a cine file in the directory.
# Mlist is a list of variables associated to each element of Slist.
Slist = Sdata_manip.load_serie(date, indices)
print 'file name -->', Slist[0].fileCine
Mlist = Sdata_manip.load_measures(Slist, indices=0)  # refer to the index of the measurement. here 0 - Stephane
Mfluc_list = Sdata_manip.load_measures(Slist, indices=0)

M = Mlist[0]
print 'M = ', M
Mfluc = Mfluc_list[0]

t = range(0, Mfluc.Ux.shape[2])
Ux_mean = np.nanmean(M.Ux[..., t], axis=2)  # Mean Ux (Time average)
Uy_mean = np.nanmean(M.Uy[..., t], axis=2)  # Mean Uy (Time average)

# Reynolds decomposition: u = U + u'
for t in range(0, Mfluc.Ux.shape[2]):
    for x in range(0, Mfluc.Ux.shape[0]):
        for y in range(0, Mfluc.Ux.shape[1]):
            # calculate the turbulent flow u'_x, u'_y
            Mfluc.Ux[x, y, t] = Mfluc.Ux[x, y, t] - Ux_mean[x, y]
            Mfluc.Uy[x, y, t] = Mfluc.Uy[x, y, t] - Uy_mean[x, y]

# Print the name of the cine file used to produce the Sdata
print Slist[0].fileCine

print 'M.Ux[12, 18, 150] = ', M.Ux[12, 18, 150]
print 'Mfluc.Ux.shape = ', Mfluc.Ux.shape
print 'Mfluc.Ux[12, 18, 150] = ', Mfluc.Ux[12, 18, 150]
print 'Ux_mean.shape = ', Ux_mean.shape
print 'Ux_mean[12, 18] ', Ux_mean[12, 18]

M = Mlist[0]
for M in Mlist:
    M.add_param('v', 'mms')
    M.add_param('freq', 'Hz')

M.get('E')
M.get('omega')
M.get('enstrophy')
M.get('dU')

Mfluc = Mfluc_list[0]
for Mfluc in Mfluc_list:
    Mfluc.add_param('v', 'mms')
    Mfluc.add_param('freq', 'Hz')

Mfluc.get('E')
Mfluc.get('omega')
Mfluc.get('enstrophy')
Mfluc.get('dU')

# M attributes
print M.shape()
print M.Uy[10, :, 0]
print M.Id.date
print M.Id.index

for M in Mlist:
    print(M.Sdata.fileCine[-20:-5])
    print(M.Sdata.fileDir)
    print(M.Sdata.filename)  # name of the file(SData) that stores measures (M)
    print(M.Sdata.param)
    print M.shape()

# Plot the time averaged energy
frame = 10
# args are (M, variable, frame, vmin, vman)
graphes.Mplot(M, 'E', frame, vmin=0, vmax=100000)
graphes.colorbar()

# Plot the vorticity at a particular time
graphes.Mplot(M, 'omega', frame, vmin=-300, vmax=300)
graphes.colorbar()

# Plot the enstrophy at a particular time (frame#)
graphes.Mplot(M, 'enstrophy', frame, vmin=0, vmax=90000)
graphes.colorbar()

########################################################
figs = {}
fields, names, vmin, vmax, labels, units = comp.std_fields()
# print(M.param.v)
indices = range(50, M.Ux.shape[2])
field = 'E'
j = 3  # index to put right units on the graph (j=3 (energy), j=4 (enstrophy))
print(fields)
E_moy = np.nanmean(getattr(M, field)[..., indices], axis=2)
graphes.color_plot(M.x, M.y, E_moy, fignum=j + 1, vmin=0, vmax=80000)
graphes.colorbar(label=names[j] + ' (' + units[j] + ')')
figs.update(graphes.legende('X (mm)', 'Y (mm)', 'Time averaged ' + field, cplot=True))

X, Y = [], []
for i in indices:
    X.append(track.positions(M, i, field='E', indices=indices, step=1, sigma=10.)[1])
    Y.append(track.positions(M, i, field='E', indices=indices, step=1, sigma=10.)[3])
X0, Y0 = np.nanmean(X), np.nanmean(Y)
# graphes.graph([X0],[Y0],label='ko',fignum=j+1)
graphes.save_figs(figs, savedir=savedir, suffix='smooth')

########################################################
# Time averaged Fluctuating Energy
########################################################
figs = {}
fields, names, vmin, vmax, labels, units = comp.std_fields()
# print(M.param.v)
indices = range(50, Mfluc.Ux.shape[2])
field = 'E'
j = 3  # index to put right units on the graph (j=3 (energy), j=4 (enstrophy))
print(fields)
E_moy = np.nanmean(getattr(Mfluc, field)[..., indices], axis=2)
graphes.color_plot(Mfluc.x, Mfluc.y, E_moy, fignum=j + 1, vmin=0, vmax=80000)
graphes.colorbar(label=names[j] + ' \'' + ' (' + units[j] + ')')
figs.update(graphes.legende('X (mm)', 'Y (mm)', 'Time averaged ' + field + '\'', cplot=True))

X, Y = [], []
for i in indices:
    X.append(track.positions(Mfluc, i, field='E', indices=indices, step=1, sigma=10.)[1])
    Y.append(track.positions(Mfluc, i, field='E', indices=indices, step=1, sigma=10.)[3])
X0, Y0 = np.nanmean(X), np.nanmean(Y)
# graphes.graph([X0],[Y0],label='ko',fignum=j+1)
graphes.save_figs(figs, savedir=savedir, suffix='smooth')

########################################################
# Time averaged vorticity
########################################################
figs = {}
fields, names, vmin, vmax, labels, units = comp.std_fields()
print(Mfluc.param.v)
print(fields)
indices = range(50, Mfluc.Ux.shape[2])
field = 'omega'

print(names)
j = 4
E_moy = np.nanmean(getattr(M, field)[..., indices], axis=2)
graphes.color_plot(M.x, M.y, E_moy, fignum=j + 1, vmin=-40, vmax=40)
graphes.colorbar(label=names[j] + ' (' + units[j] + ')')
figs.update(graphes.legende('X (mm)', 'Y (mm)', 'Time averaged ' + field, cplot=True))

X, Y = [], []
for i in indices:
    X.append(track.positions(M, i, field=field, indices=indices, step=1, sigma=10.)[1])
    Y.append(track.positions(M, i, field=field, indices=indices, step=1, sigma=10.)[3])
X0, Y0 = np.nanmean(X), np.nanmean(Y)
# graphes.graph([X0],[Y0],label='ko',fignum=j+1)
graphes.save_figs(figs, savedir=savedir, suffix='smooth')

########################################################
# Time averaged fluctuating vorticity
########################################################
figs = {}
fields, names, vmin, vmax, labels, units = comp.std_fields()
print(M.param.v)
print(fields)
indices = range(50, M.Ux.shape[2])
field = 'omega'

print(names)
j = 4
E_moy = np.nanmean(getattr(Mfluc, field)[..., indices], axis=2)
graphes.color_plot(Mfluc.x, Mfluc.y, E_moy, fignum=j + 1, vmin=-40, vmax=40)
graphes.colorbar(label=names[j] + ' \'' + ' (' + units[j] + ')')
figs.update(graphes.legende('X (mm)', 'Y (mm)', 'Time averaged ' + field + '\'', cplot=True))

X, Y = [], []
for i in indices:
    X.append(track.positions(Mfluc, i, field=field, indices=indices, step=1, sigma=10.)[1])
    Y.append(track.positions(Mfluc, i, field=field, indices=indices, step=1, sigma=10.)[3])
X0, Y0 = np.nanmean(X), np.nanmean(Y)
# graphes.graph([X0],[Y0],label='ko',fignum=j+1)
# graphes.save_figs(figs,savedir=savedir,suffix='smooth')


########################################################
# Time averaged enstrophy
########################################################
figs = {}
fields, names, vmin, vmax, labels, units = comp.std_fields()
print(M.param.v)
print(fields)
indices = range(50, 150)
field = 'enstrophy'

print(names)
j = 4
E_moy = np.sqrt(np.nanmean(getattr(M, field)[..., indices], axis=2))
graphes.color_plot(M.x, M.y, E_moy, fignum=j + 1, vmin=0, vmax=150)
graphes.colorbar(label=names[j] + ' (' + units[j] + ')')
figs.update(graphes.legende('X (mm)', 'Y (mm)', 'Time averaged ' + field, cplot=True))

X, Y = [], []
for i in indices:
    X.append(track.positions(M, i, field=field, indices=indices, step=1, sigma=10.)[1])
    Y.append(track.positions(M, i, field=field, indices=indices, step=1, sigma=10.)[3])
X0, Y0 = np.nanmean(X), np.nanmean(Y)
# graphes.graph([X0],[Y0],label='ko',fignum=j+1)
graphes.save_figs(figs, savedir=savedir, suffix='squared omega')

########################################################
# Time averaged enstrophy'
########################################################
# Why enstrophy = enstrophy' (calculated using u' field)??
figs = {}
fields, names, vmin, vmax, labels, units = comp.std_fields()
print(Mfluc.param.v)
print(fields)
indices = range(50, 150)
field = 'enstrophy'

print(names)
j = 4
E_moy = np.sqrt(np.nanmean(getattr(Mfluc, field)[..., indices], axis=2))
graphes.color_plot(Mfluc.x, Mfluc.y, E_moy, fignum=j + 1, vmin=0, vmax=150)
graphes.colorbar(label=names[j] + ' \'' + ' (' + units[j] + ')')
figs.update(graphes.legende('X (mm)', 'Y (mm)', 'Time averaged ' + field + ' \'', cplot=True))

X, Y = [], []
for i in indices:
    X.append(track.positions(Mfluc, i, field=field, indices=indices, step=1, sigma=10.)[1])
    Y.append(track.positions(Mfluc, i, field=field, indices=indices, step=1, sigma=10.)[3])
X0, Y0 = np.nanmean(X), np.nanmean(Y)
# graphes.graph([X0],[Y0],label='ko',fignum=j+1)
# graphes.save_figs(figs,savedir=savedir,suffix='squared omega')


########################################################
# Energy over time
########################################################
E_moy = np.nanmean(M.E, axis=(0, 1))
graphes.graph(M.t, E_moy, label='ko-')
# graphes.set_axis(0,5,0,18000)
figs = graphes.legende('Time (s)', 'Energy (mm$^2$/s$^2$)', '')
graphes.save_figs(figs, savedir=savedir, suffix='400mms')

########################################################
# Energy over time
########################################################
E_moy = np.nanmean(Mfluc.E, axis=(0, 1))
graphes.graph(Mfluc.t, E_moy, label='ko-')
# graphes.set_axis(0,5,0,18000)
figs = graphes.legende('Time (s)', 'Turbulent Energy (mm$^2$/s$^2$)', '')
graphes.save_figs(figs, savedir=savedir, suffix='v400mms')

########################################################
# Saves Ux,Uy,omega,E,enstrophy in 'savedir'/'date'/'date'+'indicies'+'version'/Example
########################################################
# Specify the index (Recall each index corresponds to a cine file)
# I don't think the above statement is true -- I think these are time indices (frame numbers)
# indices = 2
# figs = comp.vortex_collider(M, indices, version=1, outdir=savedir)
# graphes.plt.close('all')


########################################################
########################################################
Ux = np.nanmean(M.Ux[..., indices], axis=2)
Uy = np.nanmean(M.Uy[..., indices], axis=2)

nt = len(indices)
# Reynolds decomposition
Ux_fluct = M.Ux[..., indices] - np.transpose(np.tile(Ux, (nt, 1, 1)), (1, 2, 0))
Uy_fluct = M.Uy[..., indices] - np.transpose(np.tile(Uy, (nt, 1, 1)), (1, 2, 0))

E_fluct = Ux_fluct ** 2 + Uy_fluct ** 2
graphes.color_plot(M.x, M.y, np.nanmean(E_fluct, axis=2), vmin=0, vmax=50000)
figs = graphes.legende('$X$ (mm)', '$Y$ (mm)', 'Energy fluctuating(mm$^2$/s$^{2}$)', cplot=True)
graphes.colorbar()
graphes.save_figs(figs, savedir=savedir)

########################################################
# Creates a folder with png files of E/omega/enstrophy for each frame
########################################################
graphes.movie(Mlist[0], 'E', Dirname=savedir, vmin=0, vmax=3 * 10 ** 5)
graphes.movie(Mlist[0], 'omega', Dirname=savedir, vmin=-300, vmax=300)
graphes.movie(Mlist[0], 'enstrophy', Dirname=savedir, vmin=0, vmax=90000)

########################################################
# Energy vs time
########################################################
M = Mlist[0]
field = 'E'

fig, ax = graphes.set_fig(1, subplot=111)
fig.set_size_inches(20, 4)
Y_moy = np.nanmean(getattr(M, field), axis=(0, 1))
nx, ny, nt = M.shape()
indices = range(nt)
graphes.semilogy(M.t[indices], Y_moy[indices], label='ks-', fignum=1)
i0 = 28
T = 60
t0 = M.t[range(1, T - 9)]
print(M.t[i0])
# for i in range(nt//T-3):
#    ind = range(i0,i0+T-10)
#    graphes.graph([M.t[i0]],[Y_moy[i0]],label='rs',fignum=1)
figs = graphes.legende('Time (s)', 'Energy (mm$^2$/s$^{2}$)', '')
graphes.set_axis(0, 8, 10 ** 1, 10 ** 5)
#    graphes.graphloglog(t0,Y_moy[ind],label='.-',fignum=2)
#    i0 = i0+T
suffix = '_tmpsuffix_'
graphes.save_figs(figs, savedir=savedir + subdir, suffix=suffix + 'Evst')

graphes.graphloglog(t0, 10 ** 3 * t0 ** (-2), label='r--', fignum=2)
graphes.graphloglog(t0, 10 ** 3 * t0 ** (-1.4), label='r+--', fignum=2)
graphes.set_axis(10 ** -2, 10 ** 0, 10 ** 2, 10 ** 5)

figs.update(graphes.legende('Translated Time (s) ', 'Energy (mm$^2$/s$^{2}$)', ''))
suffix = graphes.set_name(M, param=['freq', 'v'])
graphes.save_figs(figs, savedir=savedir + subdir, suffix=suffix + 'shaped')
plt.close('all')

########################################################
# Fourier to obtain Energy spectrum in k space
########################################################
M = Mlist[0]
# compute_spectrum_2d() returns S_E, kx, ky
# compute_spectrum_1d() returns S_E, k
Fourier.compute_spectrum_2d(M, Dt=3)
Fourier.compute_spectrum_1d(M, Dt=3)
M.S_k[:, 100]

########################################################
# Plot total Energy and total Omega for each frame
########################################################
M = Mlist[0]
Enstrophy_total = np.mean(M.omega ** 2, axis=(0, 1))
Omega_total = np.mean(M.omega, axis=(0, 1))
E_total = np.mean(M.E, axis=(0, 1))

figs = {}
graphes.graph(M.t, Omega_total, fignum=1)
figs.update(graphes.legende('Time (s)', 'Vorticity $\Omega$ (s$^{-1}$)', ''))
graphes.save_figs(figs, savedir=savedir, suffix=M.Id.get_id() + '_Full_Vorticity_plot', display=True)
plt.close('all')

graphes.graph(M.t, E_total, fignum=2)
figs.update(graphes.legende('Time (s)', 'Energy $E$ (mm$^2$ s$^{-2}$)', ''))
graphes.save_figs(figs, savedir=savedir, suffix=M.Id.get_id() + '_Full_Energy_plot', display=True)
plt.close('all')

# tt = M.t[175:500];
# EE = E_total[175:500];
# oo = np.sqrt(Enstrophy_total[175:500]);
# graphes.graph(tt,EE,fignum=3)
# figs.update(graphes.legende('Time (s)','Energy $E$ (mm$^2$ s$^{-2}$)',''))
# graphes.set_axis(0.35,0.57,0,31000)
# graphes.save_figs(figs,savedir=savedir,suffix=M.Id.get_id()+'_Partial_Energy_plot',display=True)
# graphes.graph(tt,oo,fignum=4)
# figs.update(graphes.legende('Time (s)','Vorticity $\Omega$ (s$^{-1}$)',''))
# graphes.set_axis(0.35,0.57,0,80)
# graphes.save_figs(figs,savedir=savedir,suffix=M.Id.get_id()+'_Partial_Vorticity_plot',display=True)

########################################################
# Energy spectrum (Kolmogorov scaling: -5/3)
########################################################
plt.close('all')
M = Mlist[0]
for ii in range(20, 148):
    print 'Plotting energy spectrum, item =', ii
    graphes.graphloglog(M.k, M.S_k[..., ii], 'k')

graphes.graph(M.k, 15 * M.k ** (-5. / 3), label='r-')
figs = graphes.legende(r'$k$ [mm$^{-1}$]', r'$E$ [mm/s$^{2}$]', '')
graphes.save_fig(1, savedir + 'energy_spectrum1', frmt='pdf', dpi=300, overwrite=False)

########################################################
# Energy spectrum as function of shell radius
########################################################



sys.exit()
# Above the codes work nice and sound. Well-organized (run cells from the top)
# What is this? - Takumi 7/10/17
indices = 2
comp.comparison(Mlist, indices, outdir=savedir)
for i, M in enumerate(Mlist):
    Ux_moy = np.nanmean(M.Ux, axis=2)
    Uy_moy = np.nanmean(M.Uy, axis=2)
    E_moy = Ux_moy ** 2 + Uy_moy ** 2
    graphes.color_plot(M.x, M.y, E_moy, fignum=i + 1, vmin=0, vmax=0000)
    graphes.colorbar()

# Should compute the spatial decay: not working currently....
for M in Mlist:
    figs = comp.spatial_decay(M, indices=indices, outdir=savedir)
    graphes.save_figs(figs, savedir=savedir + 'Spatial_decay', prefix=graphes.set_name(M, param=['freq', 'v']))
    graphes.plt.close('all')

##
from mpl_toolkits.axes_grid.inset_locator import inset_axes

R, Theta = Smath.cart2pol(M.x - X0, M.y - Y0)

Z = np.ndarray.flatten(Z)
R_flat = np.ndarray.flatten(R)
Theta_flat = np.ndarray.flatten(Theta)

R0 = 20
phi = np.pi / 2
C = np.mod((Theta_flat + phi + np.pi) / 2 / np.pi, 1)
cmap = plt.cm.hot
color = [plt.colors.rgb2hex(cmap(c)[:3]) for c in C]

fig, ax2 = graphes.set_fig(1, subplot=122)
fig.set_size_inches(20, 6)
sc = ax2.scatter(R_flat, Z, marker='o', facecolor=color, alpha=0.3, lw=0, cmap=cmap)
# plt.plt.colorbar(sc)
# graphes.scatter([np.log10(r)],[np.log10(r)],label='o',c=c,alpha=0.6)
# plt.xscale('log')
# graphes.plt.colorbar()
# cbar = fig.colorbar(cax, ticks=[-1, 0, 1])
# graphes.graphloglog([R0,R0],[10**2,8*10**4],label='b-')
Rth = np.arange(10 ** 1, 10 ** 2, 1.)
graphes.graphloglog(Rth, 10 ** 5 * (Rth / R0) ** -3.2, label='k--')
graphes.graphloglog(Rth, 5 * 10 ** 4 * (Rth / R0) ** -4.5, label='k--')
# graphes.graphloglog(R,Z,label='ko')
graphes.set_axis(10 ** 0, 10 ** 2, 10 ** 2, 8 * 10 ** 4)
# cbar = fig.colorbar(ax1, ticks=[0, 1], orientation='vertical')
figs = graphes.legende('$R$ (mm)', 'Energy (mm$^2$/s$^{2}$)', 'Spatial decay')

fig, ax1 = graphes.set_fig(1, subplot=121)
graphes.color_plot(M.x, M.y, E_moy, fignum=1, vmin=0, vmax=40000, subplot=121)
graphes.colorbar(label=names[j] + ' (' + units[j] + ')')
figs.update(graphes.legende('X (mm)', 'Y (mm)', 'Time averaged ' + field, cplot=True))

inset_ax = inset_axes(ax2, height="50%", width="50%", loc=3)
inset_ax.pcolormesh(M.x / 10 - 10, M.y / 10 - 10, Theta, cmap=cmap)
inset_ax.axis('off')
graphes.save_figs(figs, savedir=savedir, prefix='Final', suffix='Colored_Scaling_Exponent_from_32d_to45d')

R, Theta = Smath.cart2pol(M.x - X0, M.y - Y0)

print(Z.shape)
print(R.shape)

R0 = 20
graphes.graphloglog(R, Z, label='ko')
graphes.graphloglog([R0, R0], [10 ** 2, 8 * 10 ** 4], label='b-')

graphes.graphloglog(R, 10 ** 5 * (R / R0) ** -3.2, label='r--')
graphes.graphloglog(R, 5 * 10 ** 4 * (R / R0) ** -4.5, label='r--')
# graphes.graphloglog(R,Z,label='ko')
graphes.set_axis(10 ** 0, 10 ** 2, 10 ** 2, 8 * 10 ** 4)
figs = graphes.legende('$R$ (mm)', 'Energy (mm$^2$/s$^{2}$)', 'Spatial decay')
graphes.save_figs(figs, savedir=savedir, suffix='Scaling_Exponent_from_32d_to45d')
