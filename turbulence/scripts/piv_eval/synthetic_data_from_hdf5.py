import os
import numpy as np
import glob as glob


import argparse

import sys


import ilpm.ilpm.networks as nt
import matplotlib.pyplot as plt
from PIL import Image
import h5py
from scipy.spatial import KDTree
import time

'''
Create synthetic data by forming a random array of dots, then moving them according to ux, uy, uz from a given hdf5 file
Experimental parameters
----------------------- 
density = 0.6 / 64. pixels^{-2}
diameter = 10 pixels

Options
-------
-n --npts Number of dots in the areal view 
-res -resolution Number of pixels in each dimension of the areal view (size of image)
-dt -dt timestep between images
-sz -size Size of the dots to create
-sz_method -size_method Method for size of dots ('gaussian', 'white')
-ssig -size_sigma Variation in size of dots, in units of args.size
-shape -shape_method Method for shape of dots ('gaussian', 'white') --> whether the dot is a gaussian blob or a circle
-lsig --luminosity_sigma Variation in luminosity
-lum_method --lum_method Method for initial luminosity distribution
-lumc_mag --lum_change_magnitude Max magnitude of luminosity change
-lumc_method --lum_change_method Method for changing luminosity between frames ('zplane', 'random'homon) 
-cutoff, --cutoff_sigma, Maximum distance over which particle influences luminosity 
                         in units of args.size * (1 + size_sigma)
-maxi, --max_intensity Value of summed image to set to saturation value (white)
-datadir, --datadir, Directory where HDF5 data lies

Example Usage:
python synthetic_data_from_hdf5.py -data /Volumes/labshared3-1/takumi/JHTD-sample/JHT_Database/Data/synthetic_data_from_bob/double_oseen_PIV_gamma17000_NP50000_r40_D150_cx640_cy400_fps500_w1280_h800_psize3_dx200/hdf5data/double_oseen_PIV_gamma17000_NP50000_r40_D150_cx640_cy400_fps500_w1280_h800_psize3_dx200.h5
-dt 1. -n 50000
'''


def make_image(gridpts, pts, tree, radii, lums, args, max_intensity=None, show=False):
    """Create an image of particles with individually specified radii and luminosities, colored either as discs (if
    args.shape_method == 'white') or as Gaussian blobs (if args.shape_method == 'gaussian').

    Parameters
    ----------
    gridpts : N x 2 float array
        the positions in space of the image pixels
    pts : M x 2 float array
        the particle positions in the image
    tree : KDTree instance
        the KD tree of the 2D float array of pixel locations stored in gridpts.
    radii : M x 1 float array
        the size of each particle
    lums : M x 1 float array
        the luminosity values of the particles
    args : argparse instance
        with attributes size, cutoff_sigma, size_sigma, and shape_method.
    max_intensity : float or None
        the maximum intensity of the image. If None, normalizes to the maximum of the created image
    show : bool
        display the image before return

    Returns
    -------
    im : PIL.Image instance (with uint8 intensity array)
        the output image
    """
    # Iterate over each particle, summing the intensity contribution of each
    inten = np.zeros_like(gridpts[:, 0])
    for (radius, lum, kk) in zip(radii, lums, range(np.shape(pts)[0])):
        if kk % 100 == 0:
            print 'considering particle ' + str(kk) + '...'
        inds = tree.query_ball_point(pts[kk], r=args.cutoff_sigma * float(args.size) * (1. + args.size_sigma))
        if not inds:
            #print 'pt = ', pts[kk]
            continue
            raise RuntimeError('Empty indices for particle ' + str(kk))

        dists = nt.dist_pts(gridpts[inds], np.array([pts[kk]])).ravel()
        if args.shape_method == 'gaussian':
            inten[inds] += (lum * intensity_gauss(dists, sigma=radius)).ravel()
        elif args.shape_method == 'white':
            inten += (lum * intensity_white(dists, sigma=radius)).ravel()

    # Set maximum brightness and overall scale
    if max_intensity is None:
        inten /= np.max(inten)
    else:
        inten /= max_intensity
        inten[inten > 1.] = max_intensity

    inten *= 255.
    inten = inten.reshape(np.shape(xgrid))
    im = Image.fromarray(inten.astype(np.uint8))

    if show:
        plt.imshow(im)
        plt.colorbar()
        plt.show()

    return im


def intensity_gauss(dist, sigma=1.):
    """Create contribution to the intensity of an image that is Gaussian in distance from a (bright) particle.

    Parameters
    ----------
    dist : n x m float array or nm x 1 float array
        the distances of each pixel from each particle
    sigma : float or nm x 1 float array
        the standard deviation of the Gaussian scattering pattern for each particle

    Returns
    -------
    intensity : float array of same shape as dist
        the intensity pattern from the particles
    """
    return np.exp(-dist ** 2 / sigma ** 2)


def intensity_white(dist, width=1.):
    """Create contribution to the intensity of an image that is flat with cutoff in distance from a (bright) particle.

    Parameters
    ----------
    dist : n x m float array or nm x 1 float array
        the distances of each pixel from each particle
    width : float or nm x 1 float array
        the distance cutoff of the flat scattering pattern for each particle

    Returns
    -------
    intensity : float array of same shape as dist
        the intensity pattern from the particles
    """
    inten = np.zeros_like(dist, dtype=float)
    inten[dist < width] = 1.
    return inten


parser = argparse.ArgumentParser(description='Generate an image that mimics a PIV experiment using a vel field from a given velocity field stored in hdf5.')
parser.add_argument('-check', '--check', help='Display intermediate results', action='store_true')
parser.add_argument('-overwrite', '--overwrite', help='Overwrite previous tracking results', action='store_true')
parser.add_argument('-n', '--npts', help='Number of dots in the areal view', type=int, default=1024)
parser.add_argument('-dt', '--dt', help='Timestep. The units depend on the units of the supplied velocity.', type=float, default=0.02)
parser.add_argument('-sz', '--size', help='Size of the dots to create', type=float, default=1.0)
parser.add_argument('-sz_method', '--size_method', help='Method for size of dots', type=str, default='gaussian')
parser.add_argument('-ssig', '--size_sigma', help='Variation in size of dots, in units of args.size',
                    type=float, default=1.0)
parser.add_argument('-shape', '--shape_method', help='Method for shape of dots', type=str, default='gaussian')
parser.add_argument('-lsig', '--luminosity_sigma', help='Variation in luminosity', type=float, default=0.1)
parser.add_argument('-lum_method', '--lum_method', help='Method for initial luminosity distribution',
                    type=str, default='gaussian')
parser.add_argument('-lumc_mag', '--lum_change_magnitude', help='Max magnitude of luminosity change',
                    type=float, default=0.1)
parser.add_argument('-lumc_method', '--lum_change_method', help='Method for changing luminosity',
                    type=str, default='zplane')
parser.add_argument('-laser_thickness', '--laser_thickness',
                    help='If lumc_method==uz_gauss, laser_thickness determines how particles get dimmed by out-of plane motion',
                    type=float, default=100)
parser.add_argument('-cutoff', '--cutoff_sigma',
                    help='Maximum distance over which particle influences luminosity, '
                         'in units of args.size * (1 + size_sigma)',
                    type=float, default=1.0)
parser.add_argument('-maxi', '--max_intensity', help='Value of summed image to set to saturation value (white)',
                    type=float, default=1.)
parser.add_argument('-eps', '--epsilon', help='Minimum precision for cutoff as small number', type=float, default=1e-9)
parser.add_argument('-series_num', '--series_num', help='Number of images made between t=0 and t=dt, default=2 (t=0 and t=dt)', type=int, default=2)

# HDF5 file (velocity field)
parser.add_argument('-data', '--datapath', help='Path to velocity field data (must be in hdf5)', type=str,
                    default='/Volumes/labshared3-1/takumi/JHTD-sample/JHT_Database/Data/synthetic_data_from_bob/double_oseen_PIV_gamma17000_NP50000_r40_D150_cx640_cy400_fps500_w1280_h800_psize3_dx200/hdf5data/double_oseen_PIV_gamma17000_NP50000_r40_D150_cx640_cy400_fps500_w1280_h800_psize3_dx200.h5')
parser.add_argument('-ux', '--ux', help='String used for ux in HDF5 file. Default: ux', type=str, default='ux')
parser.add_argument('-uy', '--uy', help='String used for uy in HDF5 file. Default: uy', type=str, default='uy')
parser.add_argument('-uz', '--uz', help='String used for uz in HDF5 file. Default: uz', type=str, default='uz')
parser.add_argument('-x', '--x', help='String used for x in HDF5 file. Default: x', type=str, default='x')
parser.add_argument('-y', '--y', help='String used for y in HDF5 file. Default: y', type=str, default='y')


args = parser.parse_args()

# Check if hdf5 file exists
if not os.path.exists(args.datapath):
    print 'Given HDF5 does not exist!'
    print 'Data path: ' + args.datapath
    print 'Exiting...'
    sys.exit(1)
else:
    print 'Searching ' + args.datapath
    print 'HDF5 file found!'

# Data architecture
datadir = os.path.split(args.datapath)[0]
savedir = os.path.join(datadir, 'synthetic_data/')
if not os.path.exists(savedir):
    print 'creating dir: ', savedir
    os.makedirs(savedir)

# Load a velocity field from given hdf5 file(s)
if glob.glob(args.datapath):
    with h5py.File(args.datapath, 'r') as ff:
        ux, uy = np.array(ff[args.ux]), np.array(ff[args.uy])
        if args.uz in ff.keys():
            uz = np.array(ff[args.uz])
        else:
            # If uz does not exist, let luminosity fluctuate randomly instead of using out-of-plane motion based on uz
            print 'HDF5 file does not contain uz'
            print 'Change luminosity fluctuation method to be random from z_plane'
            args.lum_change_method = 'random'

        xx, yy = np.array(ff[args.x]), np.array(ff[args.y])
        imsize = xx.shape
        ff.close()

# Prepare output directory
specstr = 'npts{0:05d}'.format(args.npts) + '_shape{0:05d}x{1:05d}'.format(imsize[0], imsize[1])
specstr += '_dt{0:0.5f}'.format(args.dt).replace('.', 'p') + '_sz{0:0.3f}'.format(args.size).replace('.', 'p')
specstr += '_' + args.size_method
specstr += '_szsig{0:0.3f}'.format(args.size_sigma).replace('.', 'p')
specstr += '_shape' + args.shape_method
specstr += '_lsig{0:0.3f}'.format(args.luminosity_sigma).replace('.', 'p')
specstr += '_' + args.lum_method
specstr += '_lsig{0:0.3f}'.format(args.lum_change_magnitude).replace('.', 'p')
specstr += '_' + args.lum_change_method
specstr += '_cutoff{0:0.3f}'.format(args.cutoff_sigma).replace('.', 'p')
specstr += '_maxi{0:0.3f}'.format(args.max_intensity).replace('.', 'p')

#
# # Create snapshot images as pngs of what we are sampling
# if not glob.glob(savedir + 'ux_z0.png'):
#     for (vel, label, name) in zip([ux, uy], [r'$U_x$', r'$U_y$'], ['ux', 'uy']):
#         plt.figure(figsize=(32, 24))
#         plt.subplot(1, 1, 1)
#         plt.pcolor(xx, yy, vel, cmap='RdBu', label='z=0')
#         cbar = plt.colorbar()
#         cbar.ax.set_ylabel(label, fontsize=75)
#         cbar.ax.tick_params(labelsize=50)
#         plt.xlabel(r'$x$', fontsize=75)
#         plt.ylabel(r'$y$', fontsize=75)
#         plt.xticks(size=50)
#         plt.yticks(size=50)
#         plt.axis('scaled')
#         filename = savedir + name + '_z0' + '.png'
#         print 'saving ', filename
#         plt.savefig(filename)
#         plt.close('all')
#
# # Look at distribution
# filename = savedir + 'velocity_distribution.png'
# if not glob.glob(filename):
#     plt.figure()
#     plt.hist2d(ux.ravel()[~np.isnan(ux.ravel())], uy.ravel()[~np.isnan(uy.ravel())], bins=100)
#     plt.xlabel(r'$U_x$')
#     plt.ylabel(r'$U_y$')
#     plt.suptitle(r'2D Velocity distribution')
#     plt.colorbar()
#     print 'saving ', filename
#     plt.savefig(filename)
#
# filename = savedir + 'velocity_distribution1d_z0_hist.png'
# if not glob.glob(filename) or True:
#     plt.figure()
#     px, xb = np.histogram(ux.ravel()[~np.isnan(ux.ravel())], bins=100)
#     py, yb = np.histogram(uy.ravel()[~np.isnan(uy.ravel())], bins=100)
#     # Look at the midpoints of the bins
#     xb = xb[:-1] + np.diff(xb) * 0.5
#     yb = yb[:-1] + np.diff(yb) * 0.5
#     plt.plot(xb, px / float(len(ux.ravel())), '.-', label=r'$U_x$')
#     plt.plot(yb, py / float(len(ux.ravel())), '.-', label=r'$U_y$')
#     plt.legend(loc='best')
#     plt.xlabel(r'$U_{x,y}$')
#     plt.ylabel(r'probability density, $p(U_{x,y})$')
#     plt.suptitle(r'Velocity distribution in ($z=0,$ $t=0$) slice')
#     print 'saving ', filename
#     plt.savefig(filename)
#     plt.close('all')


########################
# Create synthetic data
########################
t0 = time.time()
# pts = np.random.rand(args.npts, 2) * float(args.resolution)
x, y = np.random.rand(args.npts) * imsize[1], np.random.rand(args.npts) * imsize[0]
pts = np.array((zip(x, y)))

# Determine particle sizes
if args.size_method == 'gaussian':
    radii = np.random.normal(loc=args.size, scale=args.size_sigma, size=args.npts)
    radii[radii < 0] = args.epsilon
elif args.size_method == 'white':
    radii = args.size_sigma * np.random.rand(args.npts)
    radii[radii < 0] = args.epsilon
else:
    raise RuntimeError('Size method for determining the size of dots is not understood.')

# Determine luminosity
if args.lum_method == 'gaussian':
    lums = np.random.normal(loc=1.0, scale=args.luminosity_sigma, size=args.npts)
    lums[lums < 0] = args.epsilon
elif args.lum_method == 'white':
    lums = args.luminosity_sigma * np.random.rand(args.npts)
    lums[lums < 0] = args.epsilon
else:
    raise RuntimeError('Luminosity method for determining the brightness distribution is not understood.')

xgrid, ygrid = xx, yy

gridpts = np.dstack((xgrid.ravel(), ygrid.ravel()))[0]

allpts = np.vstack((gridpts, pts))
# leafsize
leafsize = args.cutoff_sigma * float(args.size) * (1. + args.size_sigma) * float(max(imsize[0], imsize[1]))
print 'leafsize = ' + str(leafsize)
leafsize = np.ceil(leafsize)
leafsize = max(6, leafsize)
print 'Leaf size for KDTree: ' + str(leafsize)
tree = KDTree(gridpts, leafsize=leafsize)
print 'Created KDTRee...'

# Create the image from the tree
im = make_image(gridpts, pts, tree, radii, lums, args, max_intensity=args.max_intensity)
im.save(savedir + specstr + '_0.png')

t1 = time.time()
print 'elapsed time = ' + str(t1 - t0) + '/n'

##################################################################
# Update the positions and luminosities and recompute an image
##################################################################
# Only use minimum distance of each particle to nearest gridpoint to sample the velocity
# Make a displacement (velocity * dt) vector for each particle.

# Let particles move for args.series_num * dt. Save if i == save.index
saveindex = [1, 10]
for i in range(1, args.series_num):
    print 'Image num.: %d / %d' % (i-1, args.series_num-1)
    inds = np.zeros_like(pts[:, 0], dtype=int)
    for (pt, ii) in zip(pts, range(len(inds))):
        if ii % 100 == 0:
            print 'finding nearest gridpt for tracer particle ' + str(ii) + ' / ' + str(len(pts))
        inds[ii] = tree.query(pt, k=1)[1]

    pts[:, 0] += ux.ravel()[inds] * args.dt
    pts[:, 1] += uy.ravel()[inds] * args.dt

    if args.lum_change_method == 'uz':
        lums += (2. * np.random.rand(len(pts)) - 1) * args.lum_change_magnitude * uz.ravel() / np.max(np.abs(uz.ravel()))
    elif args.lum_change_method == 'random':
        lums += np.random.rand(len(pts)) * args.lum_change_magnitude
    elif args.lum_change_method == 'uz_gauss':
        lums = lums * np.exp(- (uz * args.dt) ** 2 / (0.125 * args.laser_thickness))


    if i in saveindex:
        # Iterate over each particle, summing the intensity contribution of each
        im = make_image(gridpts, pts, tree, radii, lums, args, max_intensity=args.max_intensity)
        im.save(savedir + specstr + "_%d.png" % i)

    t2 = time.time()
    print 'elapsed time = ' + str(t2 - t0)