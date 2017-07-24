# Can change matplotlib backend here
# import matplotlib
# matplotlib.use('Agg')
import turbulence.display.plotting as tplt
import turbulence.mdata.Sdata_manip as Sdata_manip
import numpy as np
import turbulence.display.graphes as graphes
import turbulence.analysis.Fourier as Fourier
import matplotlib.pyplot as plt
import turbulence.display.colormaps as tcmaps
import glob
import argparse

"""Test out how to extract E(k) for different shells of the blob"""

parser = argparse.ArgumentParser(description='Specify time string (timestr) for gyro simulation.')
parser.add_argument('-display', '--display', help='Show intermediate results in plots', action='store_true')
args = parser.parse_args()

# test masking code
# a = np.random.rand(24, 36, 3)
# b = np.zeros((24, 36))
# b[6:23, 1:34] = 1
# tplt.plot_real_matrix(b, show=True, close=True)
# d = b[:, :, np.newaxis]
# c = a * d  # b.reshape((np.shape(b)[0], np.shape(b)[1], 1))
# tplt.plot_real_matrix(c[:, :, 0], show=True, close=True, cmap='coolwarm')
# tplt.plot_real_matrix(c, show=True, close=True, cmap='coolwarm')
# print 'c= ', c
# sys.exit()


#
# radius = 15
# xx = np.arange(50) - 25
# yy = np.arange(40) - 20
# xx, yy = np.meshgrid(xx, yy)
# print 'np.shape(xx) = ', np.shape(xx)
# print 'np.shape(yy) = ', np.shape(yy)
# data = np.random.rand(50, 50, 37)
# cols = np.unique(np.where(np.abs(xx) < radius)[1])
# rows = np.unique(np.where(np.abs(yy) < radius)[0])
# include = np.logical_and(np.abs(xx) < radius, np.abs(yy) < radius)
#
# # check include
# incint = np.zeros_like(include, dtype=float)
# incint[include] = 1
# tplt.plot_real_matrix(incint, show=True, close=False)
#
# print 'rows = ', rows
# print 'cols = ', cols
# print 'include = ', include
# print 'np.shape(include) = ', np.shape(include)
# print 'np.shape(data) = ', np.shape(data)
#
# # for ind in range(np.shape(data)[2]):
# mdatr = data[np.min(rows):np.max(rows), np.min(cols):np.max(cols), :]
#
# # mdatr = data[rows, cols, :].reshape((len(rows), len(cols), -1))
# print 'np.shape(mdatr) = ', np.shape(mdatr)
#
# import turbulence.display.plotting as tplt
#
# toview = [0, 20]
# for ind in toview:
#     frame = mdatr[:, :, ind]
#     print 'np.shape(frame) = ', np.shape(frame)
#     tplt.plot_real_matrix(frame, show=True, close=True)
#
# sys.exit()


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
localroot = '/Users/npmitchell/Desktop/data_local/vortex_collision/testdata/'
savedir = localroot.replace('testdata/', 'output/') + date + '/'
rootdir = localroot + date + '/'
subdir = 'summary/'

indices = [2]  # range(6,10)
# load the Sdata from 0 to 5. If you want other datam just change the range.
# Check with the file name by typing slist[0].fileCine

# slist is Sdata_date_index_ver.hdf5.
# Each index of slist is associated to a cine file in the directory.
# mlist is a list of variables associated to each element of slist.
slist = Sdata_manip.load_serie(date, indices, rootdir=rootdir)
print 'fourier_shells_test.py: slist = ', slist
print 'fourier_shells_test.py: ss.fileCine -->', slist[0].fileCine

# if fileCine does not exist, replace the labshared2 path with the local path --> this is not general yet
for ss in slist:
    if not glob.glob(ss.fileCine):
        print 'fourier_shells_test.py: ss.fileCine -->', ss.fileCine
        ss.fileCine = ss.fileCine.replace('/Volumes/labshared2/Stephane/', localroot)
        ss.dirCine = ss.dirCine.replace('/Volumes/labshared2/Stephane/', localroot)
        print 'fourier_shells_test.py: ss.fileCine -->', ss.fileCine

print 'fourier_shells_test.py: ss.fileCine -->', slist[0].fileCine

# refer to the index of the measurement. here 0 - Stephane
mlist = Sdata_manip.load_measures(slist, indices=0)
mfluc_list = Sdata_manip.load_measures(slist, indices=0)

mm = mlist[0]
print 'mm = ', mm
mfluc = mfluc_list[0]

t = range(0, mfluc.Ux.shape[2])
Ux_mean = np.nanmean(mm.Ux[..., t], axis=2)  # Mean Ux (Time average)
Uy_mean = np.nanmean(mm.Uy[..., t], axis=2)  # Mean Uy (Time average)

# Reynolds decomposition: u = U + u'
print 'Perform Reynolds decomposition...'
for t in range(0, mfluc.Ux.shape[2]):
    for x in range(0, mfluc.Ux.shape[0]):
        for y in range(0, mfluc.Ux.shape[1]):
            # calculate the turbulent flow u'_x, u'_y
            mfluc.Ux[x, y, t] = mfluc.Ux[x, y, t] - Ux_mean[x, y]
            mfluc.Uy[x, y, t] = mfluc.Uy[x, y, t] - Uy_mean[x, y]

# Print the name of the cine file used to produce the Sdata
print slist[0].fileCine

print 'mm.Ux[12, 18, 150] = ', mm.Ux[12, 18, 150]
print 'mfluc.Ux.shape = ', mfluc.Ux.shape
print 'mfluc.Ux[12, 18, 150] = ', mfluc.Ux[12, 18, 150]
print 'Ux_mean.shape = ', Ux_mean.shape
print 'Ux_mean[12, 18] ', Ux_mean[12, 18]

mm = mlist[0]
for mm in mlist:
    mm.add_param('v', 'mms')
    mm.add_param('freq', 'Hz')

mm.get('E')
mm.get('omega')
mm.get('enstrophy')
mm.get('dU')

mfluc = mfluc_list[0]
for mfluc in mfluc_list:
    mfluc.add_param('v', 'mms')
    mfluc.add_param('freq', 'Hz')

mfluc.get('E')
mfluc.get('omega')
mfluc.get('enstrophy')
mfluc.get('dU')

# mm attributes
print mm.shape()
print mm.Uy[10, :, 0]
print mm.Id.date
print mm.Id.index

print 'mlist = ', mlist
ind = 0
for mm in mlist:
    print('properties of mlist[' + str(ind) + ']:')
    print(mm.Sdata.fileDir)
    print(mm.Sdata.filename)  # name of the file(SData) that stores measures (mm)
    print(mm.Sdata.param)
    print mm.shape()
    ind += 1

########################################################
# Fourier to obtain Energy spectrum in k space
########################################################
mm = mlist[0]
# compute_spectrum_2d() returns S_E, kx, ky
# compute_spectrum_1d() returns S_E, k
Fourier.compute_spectrum_2d(mm, Dt=3)
Fourier.compute_spectrum_1d(mm, Dt=3)

########################################################
# Energy spectrum (Kolmogorov scaling: -5/3)
# ########################################################
# plt.close('all')
# mm = mlist[0]
# for ii in range(20, 148):
#     print 'Plotting energy spectrum, item =', ii
#     graphes.graphloglog(mm.k, mm.S_k[..., ii], 'k')
#
# graphes.graph(mm.k, 15 * mm.k ** (-5. / 3), label='r-')
# figs = graphes.legende(r'$k$ [mm$^{-1}$]', r'$E$ [mm/s$^{2}$]', '')
# graphes.save_fig(1, savedir + 'energy_spectrum1', frmt='pdf', dpi=300, overwrite=False)


########################################################
# Energy spectrum as function of shell radius
########################################################
maxdim = max(np.shape(mm.x))
radii = np.arange(int(maxdim * 0.2), maxdim + 3, 2)
sk_disc = []
k_disc = []
for rad in radii:
    sk, k = Fourier.compute_spectrum_1d_within_region(mm, radius=rad, polygon=None, display=args.display, dt=10)
    sk_disc.append(sk)
    k_disc.append(k)

plt.close('all')
mm = mlist[0]
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

    graphes.graph(mm.k, 15 * mm.k ** (-5. / 3), label='r-')
    figs = graphes.legende(r'$k$ [mm$^{-1}$]', r'$E$ [mm/s$^{2}$]', r'$S(k)$ for $r <$' + str(radii[ind]) + ' pixels')
    figname = savedir + 'energy_spectrum_radius{0:03d}'.format(radii[ind])
    print 'saving fig: ' + figname
    graphes.save_fig(1, figname, frmt='pdf', dpi=300, overwrite=True)
    plt.clf()

    ind += 1

