import os
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt
from PIL import Image
import h5py
from scipy.spatial import KDTree
import time

'''
Create synthetic data by forming a random array of dots, then moving them according to ux, uy, uz from a given hdf5 file
The given hdf5 file must contain velocity values in 4D (t, z, y, x).
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
def dist_pts(pts, nbrs, dist_dim=None, square_norm=False):
    """
    Compute the distance between all pairs of two sets of points, returning an array of distances, in an optimized way.

    Parameters
    ----------
    pts: N x dim array (float or int)
        points to measure distances from
    nbrs: M x dim array (float or int)
        points to measure distances to
    dist_dim: int (default None)
        dimension along which to measure distance. Default is None, which measures the Euclidean distance
    square_norm: bool
        Abstain from taking square root. Defualt is False, so that it returns distance.

    Returns
    -------
    dist : N x M float array
        (i,j)-th element is distance between pts[i] and nbrs[j], along dimension specified (default is normed distance)
    """
    if type(pts)!='numpy.ndarray':
        pts = np.asarray(pts)
    if type(nbrs)!='numpy.ndarray':
        nbrs = np.asarray(nbrs)
    dim = pts.shape[-1]
    dist2 = np.zeros((len(pts), len(nbrs)))

    for i in range(dim):
        if dist_dim is None:
            Xarr = np.ones((len(pts), len(nbrs)), dtype=float) * nbrs[:, i]
            dist_x = Xarr - np.dstack(np.array([pts[:, i].tolist()] * np.shape(Xarr)[1]))[0]
            dist2 += dist_x ** 2
        else:
            if i == dist_dim:
                Xarr = np.ones((len(pts), len(nbrs)), dtype=float) * nbrs[:, i]
                dist_x = Xarr - np.dstack(np.array([pts[:, i].tolist()] * np.shape(Xarr)[1]))[0]
                return dist_x
    dist = np.sqrt(dist2)

    if square_norm:
        return dist2
    else:
        return dist


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
            print 'pt = ', pts[kk]
            continue
            # raise RuntimeError('Empty indices for particle ' + str(kk))

        dists = dist_pts(gridpts[inds], np.array([pts[kk]])).ravel()
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

def make_image3d(gridpts, pts, tree, radii, lums, args, max_intensity=None, show=False):

    # Iterate over each particle, summing the intensity contribution of each
    inten = np.zeros_like(gridpts[:, 0]).astype(float)
    for (radius, lum, kk) in zip(radii, lums, range(np.shape(pts)[0])):
        if kk % 100 == 0:
            print 'considering particle ' + str(kk) + '...'
        inds = tree.query_ball_point(pts[kk], r=args.cutoff_sigma * float(args.size) * (1. + args.size_sigma))
        if not inds:
            print 'pt = ', pts[kk]
            continue
            # raise RuntimeError('Empty indices for particle ' + str(kk))

        dists = dist_pts(gridpts[inds], np.array([pts[kk]])).ravel()
        if args.shape_method == 'gaussian':
            inten[inds] += (lum * intensity_gauss(dists, sigma=radius)).ravel()
        elif args.shape_method == 'white':
            inten += (lum * intensity_white(dists, sigma=radius)).ravel()


    # Reshape inten from 1D to 3D
    inten = inten.reshape(np.shape(xgrid))
    # Sum intensity along z-direction
    inten = np.sum(inten, axis=0)
    # Set maximum brightness and overall scale
    if max_intensity is None:
        inten /= np.max(inten)
        inten *= 255
    else:
        inten[inten > max_intensity] = max_intensity
    # inten *= 255

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
parser.add_argument('-pbc', '--pbc', help='Apply a periodic boundary condition', action='store_true')

# Image configurations/orientations
parser.add_argument('-z', '--z', help='z-position of the illuminated plane (x-y plane). Default is middle.', type=float, default=None)
parser.add_argument('-lt', '--laser_thickness',
                    help='If lumc_method==uz_gauss, laser_thickness determines how particles get dimmed by out-of plane motion. (px)',
                    type=float, default=10)

# Image qualities
parser.add_argument('-n', '--npts', help='Number of dots in the areal view', type=int, default=1024)
parser.add_argument('-zeta', '--zeta', help='Only velocity data will be multiplied by this factor. Note this will the state of turbulence. default: 1', type=float, default=1.0) # coarse
parser.add_argument('-nstep', '--nstep', help='No. of DNS steps between adjacent data in h5. Usually, this is 10 unless you specified tstep during pyJHTDB.cutout(...)', type=int, default=10) # coarse
parser.add_argument('-sz', '--size', help='Size of the dots to create', type=float, default=1.5)
parser.add_argument('-sz_method', '--size_method', help='Method for size of dots', type=str, default='gaussian')
parser.add_argument('-ssig', '--size_sigma', help='Variation in size of dots, in units of args.size',
                    type=float, default=0.5)
parser.add_argument('-shape', '--shape_method', help='Method for shape of dots', type=str, default='gaussian')
parser.add_argument('-lsig', '--luminosity_sigma', help='Variation in luminosity', type=float, default=0.1)
parser.add_argument('-lum_method', '--lum_method', help='Method for initial luminosity distribution',
                    type=str, default='zplane')
parser.add_argument('-lum_mag', '--lum_magnitude', help='Max magnitude of luminosity',
                    type=float, default=255)

parser.add_argument('-lumc_mag', '--lum_change_magnitude', help='Max magnitude of luminosity change',
                    type=float, default=0.1)
parser.add_argument('-lumc_method', '--lum_change_method', help='Method for changing luminosity',
                    type=str, default='zplane')
parser.add_argument('-cutoff', '--cutoff_sigma',
                    help='Maximum distance over which particle influences luminosity, '
                         'in units of args.size * (1 + size_sigma)',
                    type=float, default=1.0)
parser.add_argument('-maxi', '--max_intensity', help='Maximum intensity allowed in a pixel. Default: 255',
                    type=float, default=255)
parser.add_argument('-eps', '--epsilon', help='Minimum precision for cutoff as small number', type=float, default=1e-9)
parser.add_argument('-series_num', '--series_num', help='Number of images made between t=0 and t=dt, default=2 (t=0 and t=dt)', type=int, default=2)

# HDF5 file (velocity field)
parser.add_argument('-data', '--datapath', help='Path to velocity field data (must be in hdf5)', type=str,
                    default='/Volumes/bigraid/takumi/turbulence/JHTD/isotropic1024coarse_t0_0_tl_100_x0_0_xl_1024_y0_0_yl_1024_z0_462_zl_101_tstep_1_xstep_1_ystep_1_zstep_1.h5')


args = parser.parse_args()

# Check if hdf5 file exists
if not os.path.exists(args.datapath):
    print 'Given HDF5 does not exist!'
    print 'Data path: ' + args.datapath
    print 'Exiting...'
    sys.exit(1)
else:
    print 'Searching ' + args.datapath
    print '... HDF5 file found!'

# Data architecture
datadir = os.path.split(args.datapath)[0]
savedir = os.path.join(datadir, 'synthetic_data/')
if not os.path.exists(savedir):
    print 'Creating dir: ', savedir
    os.makedirs(savedir)

# JHTD data parameters
# ... JHTD data contains vel. field in (unit length / unit time)
fx = 2 * np.pi / 1024 # unit length / px
dt_dns = 0.0002 # unit time / dns step


# Open data
with h5py.File(args.datapath, 'r') as ff:
    # Get keys
    keys = ff.keys()
    # Get keys related to velocities
    vel_keys = [s for s in keys if s.startswith('u')]
    # Sort velocity-related keys
    vel_keys.sort()
    tdim = len(vel_keys)
    # Spatial dimension of the given velocity field
    zdim, ydim, xdim, comps = ff[vel_keys[0]].shape
    # Position of an illuminated plane
    if args.z is None:
        zpos = (zdim-1)/2
    else:
        zpos = args.z
    # Dimension of image that will be generated
    imsize = (ydim, xdim)
    # Initialize variables
    # uz, uy, ux = np.zeros((len(vel_keys), zdim, ydim, xdim)), np.zeros((len(vel_keys), zdim, ydim, xdim)), np.zeros((len(vel_keys), zdim, ydim, xdim))
    # for i, vel_key in enumerate(vel_keys):
    #     tt = float(vel_key[1:]) / 10
    #     uz[i, ...], uy[i, ...], ux[i, ...] = ff[vel_key][..., 0], ff[vel_key][..., 1], ff[vel_key][..., 2]




##########
    # INITIAL IMAGE
##########
    # Prepare an output directory name
    specstr = 'npts{0:07d}'.format(args.npts) + '_shape{0:05d}x{1:05d}'.format(imsize[0], imsize[1])
    specstr += '_z{0}'.format(zpos) + '_lthickness{0:0.2f}'.format(args.laser_thickness).replace('.', 'p')
    specstr += '_nsteps{0:03d}'.format(args.nstep) + '_zeta{0:0.2f}'.format(args.zeta).replace('.', 'p')
    specstr += '_sz{0:0.3f}'.format(args.size).replace('.', 'p') + '_' + args.size_method
    specstr += '_szsig{0:0.3f}'.format(args.size_sigma).replace('.', 'p')
    specstr += '_shape' + args.shape_method
    specstr += '_lsig{0:0.3f}'.format(args.luminosity_sigma).replace('.', 'p')
    specstr += '_' + args.lum_method
    specstr += '_lsig{0:0.3f}'.format(args.lum_change_magnitude).replace('.', 'p')
    specstr += '_' + args.lum_change_method
    specstr += '_cutoff{0:0.3f}'.format(args.cutoff_sigma).replace('.', 'p')
    specstr += '_maxi{0:0.3f}'.format(args.max_intensity).replace('.', 'p')
    specstr += '_pbc' + str(args.pbc)



    t0 = time.time()
    # Synthetic image
    # Particle positions
    z, y, x = np.random.rand(args.npts) * zdim, np.random.rand(args.npts) * ydim, np.random.rand(args.npts) * xdim
    pts = np.array((zip(z, y, x)))

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
    if args.lum_method == 'zplane':
        # luminosity is determined by z position.
        lums = args.lum_magnitude * np.exp(- 8 * (pts[:, 0] - zpos) ** 2 / (args.laser_thickness ** 2))
    elif args.lum_method == 'gaussian':
        lums = np.random.normal(loc=1.0, scale=args.luminosity_sigma, size=args.npts)
        lums[lums < 0] = args.epsilon
    elif args.lum_method == 'white':
        lums = args.luminosity_sigma * np.random.rand(args.npts)
        lums[lums < 0] = args.epsilon
    else:
        raise RuntimeError('Luminosity method for determining the brightness distribution is not understood.')
    # Show intensity distribution
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
    # a = np.linspace(-10 + zpos, 10 + zpos)
    # ax1.plot(a, args.lum_magnitude * np.exp(- 8 * (a - zpos) ** 2 / (args.laser_thickness ** 2)))
    # ax1.axvline(zpos - 2.**(-1.5)*args.laser_thickness)
    # ax1.axvline(zpos + 2.**(-1.5)*args.laser_thickness)
    # ax2.hist(lums, bins=255)
    # plt.show()

    # grid points
    # meshgrid: (Search numpy for more details)
    # Order (y, z, x) might look strange but this is how meshgrid works to create (z, y, x) grids
    ygrid, zgrid, xgrid = np.meshgrid(np.arange(ydim), np.arange(zdim), np.arange(xdim))
    gridpts = np.dstack((zgrid.ravel(), ygrid.ravel(), xgrid.ravel()))[0]
    # all points (grid points and particle positions)
    allpts = np.vstack((gridpts, pts))

    # leafsize
    leafsize = args.cutoff_sigma * float(args.size) * (1. + args.size_sigma) * float(max(xdim, ydim, zdim))
    print 'leafsize = ' + str(leafsize)
    leafsize = np.ceil(leafsize)
    leafsize = max(6, leafsize)
    print 'Leaf size for KDTree: ' + str(leafsize)
    tree = KDTree(gridpts, leafsize=leafsize)
    print 'Created KDTRee...'



    # Create the image from the tree
    im = make_image3d(gridpts, pts, tree, radii, lums, args, max_intensity=args.max_intensity)
    im.save(savedir + specstr + '_0.png')

    t1 = time.time()
    print 'elapsed time = ' + str(t1 - t0) + '/n'

    ##################################################################
    # Update the positions and luminosities and recompute an image
    ##################################################################
    # Only use minimum distance of each particle to nearest gridpoint to sample the velocity
    # Make a displacement (velocity * dt) vector for each particle.

    for j, vel_key in enumerate(vel_keys):
        # Velocities
        uz, uy, ux = ff[vel_key][..., 0], ff[vel_key][..., 1], ff[vel_key][..., 2] # length unit / time unit
        # convert to px/ dns step... zeta is an artificial constant to scale velocity
        uz = uz / fx * dt_dns * args.zeta
        uy = uy / fx * dt_dns * args.zeta
        ux = ux / fx * dt_dns * args.zeta

        # Let particles move for args.series_num * dt. Save if i == save.index
        print 'Image num.: %d / %d' % (j+1, tdim)
        inds = np.zeros_like(pts[:, 0], dtype=int)
        for (pt, ii) in zip(pts, range(len(inds))):
            if ii % 100 == 0:
                print 'finding nearest gridpt for tracer particle ' + str(ii) + ' / ' + str(len(pts))
            inds[ii] = tree.query(pt, k=1)[1]

        # pts[:, 0] += uz.ravel()[inds] * args.nstep
        pts[:, 1] += uy.ravel()[inds] * args.nstep
        pts[:, 2] += ux.ravel()[inds] * args.nstep

        # Periodic boundary conditions
        if args.pbc:
            pts[:, 0][pts[:, 0] > zdim] = pts[:, 0][pts[:, 0] > zdim] - zdim
            pts[:, 0][pts[:, 0] < 0] = pts[:, 0][pts[:, 0] < 0] + zdim
            pts[:, 1][pts[:, 1] > ydim] = pts[:, 1][pts[:, 1] > ydim] - ydim
            pts[:, 1][pts[:, 1] < 0] = pts[:, 1][pts[:, 1] < 0] + ydim
            pts[:, 2][pts[:, 2] > xdim] = pts[:, 2][pts[:, 2] > xdim] - xdim
            pts[:, 2][pts[:, 2] < 0] = pts[:, 2][pts[:, 2] < 0] + xdim


        if args.lum_change_method == 'uz':
            lums += (2. * np.random.rand(len(pts)) - 1) * args.lum_change_magnitude * uz.ravel() / np.max(np.abs(uz.ravel()))
        elif args.lum_change_method == 'random':
            lums += np.random.rand(len(pts)) * args.lum_change_magnitude
        elif args.lum_change_method == 'zplane':
            lums = args.lum_magnitude * np.exp(- 8 * (pts[:, 0] - zpos) ** 2 / (args.laser_thickness ** 2))


        # Iterate over each particle, summing the intensity contribution of each
        im = make_image3d(gridpts, pts, tree, radii, lums, args, max_intensity=args.max_intensity)
        im.save(savedir + specstr + '_%d.png' % (j+1))

        t2 = time.time()
        print 'elapsed time = ' + str(t2 - t0)