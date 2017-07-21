import numpy as np
import turbulence.display.plotting as tplt
import matplotlib.pyplot as plt
import turbulence.analysis.Fourier as four
import turbulence.mdata.Sdata_manip as Sdata_manip
import numpy as np
import turbulence.display.graphes as graphes
import turbulence.vortex.track as track
import turbulence.analysis.Fourier as Fourier
import turbulence.analysis.compilation as comp
import matplotlib.pyplot as plt
import turbulence.display.colormaps as tcmaps

"""Test out how to extract E(k) for different shells of the blob"""

import matplotlib

matplotlib.use('Agg')

test_single = False

if test_single:
    # First do single image
    radius = 30.
    a = np.random.rand(100, 100)
    sa = np.abs(np.fft.fftn(a, axes=(0, 1)))
    sa = np.fft.fftshift(sa, axes=(0, 1))
    # now mask object
    xx, yy = np.meshgrid(np.arange(100), np.arange(100))
    xx = xx - np.median(xx)
    yy = yy - np.median(yy)
    include = np.sqrt(xx ** 2 + yy ** 2) < radius
    anew = np.zeros_like(a)
    anew[include] = a[include]

    print 'np.shape(anew) = ', np.shape(anew)
    sb = np.abs(np.fft.fftn(anew, axes=(0, 1)))
    sb = np.fft.fftshift(sa, axes=(0, 1))
    tplt.plot_real_matrix(sa, show=True, climv=(-100, 100), name='F(full image)')
    tplt.plot_real_matrix(sb, show=True, climv=(-100, 100), name='F(clipped image)')
    tplt.plot_real_matrix(sb - sa, show=True, climv=(-100, 100), name='F(clipped image) - F(full image)')

# Now do time series of images
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

########################################################
# Fourier to obtain Energy spectrum in k space
########################################################
M = Mlist[0]
# compute_spectrum_2d() returns S_E, kx, ky
# compute_spectrum_1d() returns S_E, k
Fourier.compute_spectrum_2d(M, Dt=3)
Fourier.compute_spectrum_1d(M, Dt=3)

########################################################
# Energy spectrum (Kolmogorov scaling: -5/3)
# ########################################################
# plt.close('all')
# M = Mlist[0]
# for ii in range(20, 148):
#     print 'Plotting energy spectrum, item =', ii
#     graphes.graphloglog(M.k, M.S_k[..., ii], 'k')
#
# graphes.graph(M.k, 15 * M.k ** (-5. / 3), label='r-')
# figs = graphes.legende(r'$k$ [mm$^{-1}$]', r'$E$ [mm/s$^{2}$]', '')
# graphes.save_fig(1, savedir + 'energy_spectrum1', frmt='pdf', dpi=300, overwrite=False)


########################################################
# Energy spectrum as function of shell radius
########################################################
radii = [40, 80, 160, 320, 640, 1280, 2560]
sk_disc = []
k_disc = []
for rad in radii:
    sk, k = Fourier.compute_spectrum_1d_within_region(M, radius=rad, polygon=None, display=False, dt=10)
    sk_disc.append(sk)
    k_disc.append(k)

plt.close('all')
M = Mlist[0]
ind = 0
for skk in sk_disc:
    kk = k_disc[ind]

    print 'np.shape(skk) = ', np.shape(skk)
    print 'np.shape(kk) = ', np.shape(kk)
    todo = range(20, np.shape(skk)[1])
    colors = tcmaps.cubehelix_palette(n_colors=len(todo), start=0.5, rot=-0.75, check=False)
    print 'np.shape(colors) = ', np.shape(colors)
    dmyi = 0
    for ii in todo:
        print 'Plotting energy spectrum, item =', ii
        graphes.graphloglog(kk, skk[..., ii], 'k', color=colors[dmyi])
        dmyi += 1

    ind += 1

    graphes.graph(M.k, 15 * M.k ** (-5. / 3), label='r-')
    figs = graphes.legende(r'$k$ [mm$^{-1}$]', r'$E$ [mm/s$^{2}$]', r'$S(k)$ for $r <$' + str(radii[ind]) + ' pixels')
    figname = savedir + 'energy_spectrum_radius{0:03d}'.format(radii[ind])
    print 'saving fig: ' + figname
    graphes.save_fig(1, figname, frmt='pdf', dpi=300, overwrite=True)
    plt.clf()

