import scipy 
import numpy as np
import matplotlib.pyplot as plt
import basics.dataio as dio
import basics.stringformat as sf
import glob
import mayavi.mlab as mlab
from scipy.interpolate import RegularGridInterpolator
import cPickle as pkl
import argparse
from tvtk.api import tvtk

"""
Note: must be run with mayavi, so done in Canopy, for example.
"""

parser = argparse.ArgumentParser('Evolve a spring+gyro system on GPU forward then conjugate and reverse')
parser.add_argument('-overwrite', '--overwrite', 
                    help='overwrite the existing interpolation',
                    action='store_true')
parser.add_argument('-method', '--method', 
                    help='index for method from list (scatter, isosurfaces, or colorgrid) -> (0, 1, 2)',
                    type=int, default=0)
parser.add_argument('-dx', '--dx', help='Grid spacing in mm for image',
                    type=float, default=5.0)
parser.add_argument('-res', '--resolution', help='Resolution in dpi for image',
                    type=int, default=600)
args = parser.parse_args()


def sz_from_resolution(resolution=300):
    size_mm = 180.0
    size = (int(size_mm * resolution), int(size_mm * resolution))
    return size
    
# Set method as scatter, isosurfaces, or colorgrid
method_index = args.method
method = ['isosurfaces', 'scatter', 'colorgrid'][method_index]
overwrite = args.overwrite
# Note sqrt(L^2 + r^2) sin theta = r. We supply L here:
#                     r
#                    ____
#                    \  |
#                     \ |  L
#  halfangle_swept     \|
#                       .
#

# Define the distance of laser to center, in mm
dx = args.dx
ll = 241.242038
halfangle_swept = 6.0 / 180. * np.pi
angle_swept = 2. * halfangle_swept
# Assume that slices span an angle (2 * Dtheta)
nslices = 17
dtheta = angle_swept / float(nslices - 1)
# Define width of the FOV
pix2mm = 1.0

seriesdir = '/Users/npmitchell/Dropbox/Soft_Matter/turbulence/stacks_3d/small_tank_t0/'

# If the interpolation object does not exist, create it here
interpfn = seriesdir + 'slices_interp3d_dx' + sf.float2pstr(dx) + '.pkl'
fnfig = seriesdir + 'slices_interp3d_dx' + sf.float2pstr(dx) 
fnfig = seriesdir + 'slices_interp3d_dx' + sf.float2pstr(dx) 
if not glob.glob(interpfn) or overwrite:
    slices = sorted(dio.find_subdirs('slice*', seriesdir))
    for kk in range(len(slices)):
        slicedir = dio.find_subdirs('slice' + str(kk), seriesdir)[0]
        print 'examining slice ' + str(kk) 
        print 'located at: ', slicedir
        fn = glob.glob(slicedir + '*.txt')[0]
        xye = np.loadtxt(fn, skiprows=1, dtype=float)
        xy = xye[:, 0:2]
        # Update the angle of the laser
        theta = - halfangle_swept + float(kk) * dtheta
        # # define rr, which is the distance along z at the center
        # rr = np.sqrt(ll ** 2 * np.sin(theta) / (1 - np.sin(theta)))
        # hypotenuse = np.sqrt(ll ** 2 + rr ** 2)
    
        # Obtain z component from x component
        # z = hyp(x) * acos(x/hyp(x))
        xx, yy = xy[:, 0] * pix2mm, xy[:, 1] * pix2mm
        hypx = (xx + ll) / np.cos(theta)
        zz = hypx * np.sin(theta)
        xyz = np.dstack((xx, zz, yy))[0]
        
        if kk == 0:
            # get approximate maxextent for defining empty grid
            # frame_extent is the width of the frame in mm
            frame_extent = (np.max(xy.ravel()) - np.min(xy.ravel())) * pix2mm
            xextent = np.max(xx) - np.min(xx)
            yextent = np.max(yy) - np.min(yy)
            zextent = angle_swept * (ll + np.max(np.abs(xx)))
            maxextent = max(angle_swept * ll, frame_extent)
            gxarr = 0.5 * np.arange(-xextent, xextent + dx, dx)
            gyarr = 0.5 * np.arange(-yextent, yextent + dx, dx)
            gzarr = 0.5 * np.arange(-zextent, zextent + dx, dx)
            # take x->x, y->z, z->y
            xgrid, ygrid, zgrid = np.meshgrid(gxarr, gzarr, gyarr)
            points = xyz
            values = xye[:, 2]
        else:
            points = np.vstack((points, xyz))
            values = np.hstack((values, np.nan_to_num(xye[:, 2])))
    
        # print 'points = ', points
        # print 'xyz = ', xyz
        # print 'np.where(np.isnan(points)) = ', np.where(np.isnan(points))
    
    # Interpolate
    print 'interpolating data...'
    # take x->x, y->z, z->y
    egrid = scipy.interpolate.griddata(points, values, (xgrid, ygrid, zgrid), method='linear')
    print 'np.shape(egrid) = ', np.shape(egrid)
    print 'done interpolating.'
    
    # Save the interpolation
    with open(interpfn, 'wb') as fn:
        pkl.dump(egrid, fn)
else:
    # Load the interpolation
    with open(interpfn, 'rb') as fn:
        egrid = pkl.load(fn)

# Plot the result
if method == 'scatter':
    # Draw the energy as scatterplot
    plt.scatter3d()
elif method == 'isosurfaces':
    print 'plotting isosurfaces...'
    maxe = np.max(egrid.ravel())
    levels = [maxe, 0.5 * maxe, 0.25 * maxe] 
    
    size = sz_from_resolution(args.resolution)
    print 'size = ', size
    fig = mlab.figure(fgcolor=(0,0,0), bgcolor=(1,1,1), size=size)
    mlab.contour3d(egrid, transparent=True) # , contours=levels)
    
    # exp = tvtk.GL2PSExporter(file_format='pdf', sort='bsp', compress=1)
    # fig.scene.save_gl2ps(fnfig + '.pdf', exp)

    # Sideview
    # mlab.view(45, 70, 100.0)
    mlab.savefig(fnfig + '_' + str(int(args.resolution)) + '.png', magnification='auto') 
    # size = sz_from_resolution(600)
    # mlab.savefig(fnfig + '_600.png', magnification='auto', size=size)
    # mlab.savefig(fnpdf, size=None, magnification='auto')



# contour3d(scalars, ...)
# contour3d(x, y, z, scalars, ...)
# scalars is a 3D numpy arrays giving the data on a grid.
# 
# If 4 arrays, (x, y, z, scalars) are passed, the 3 first arrays give the position of the arrows, and the last the scalar value. The x, y and z arrays are then supposed to have been generated by numpy.mgrid, in other words, they are 3D arrays, with positions lying on a 3D orthogonal and regularly spaced grid with nearest neighbor in space matching nearest neighbor in the array. The function builds a scalar field assuming the points are regularly spaced.
# 
# If 4 positional arguments, (x, y, z, f) are passed, the last one can also be a callable, f, that returns vectors components (u, v, w) given the positions (x, y, z).
# 
# Keyword arguments:
# 
# color:	the color of the vtk object. Overides the colormap, if any, when specified. This is specified as a triplet of float ranging from 0 to 1, eg (1, 1, 1) for white.
# colormap:	type of colormap to use.
# contours:	Integer/list specifying number/list of contours. Specifying a list of values will only give the requested contours asked for.
# extent:	[xmin, xmax, ymin, ymax, zmin, zmax] Default is the x, y, z arrays extent. Use this to change the extent of the object created.
# figure:	Figure to populate.
# line_width:	The width of the lines, if any used. Must be a float. Default: 2.0
# name:	the name of the vtk object created.
# opacity:	The overall opacity of the vtk object. Must be a float. Default: 1.0
# reset_zoom:	Reset the zoom to accomodate the data newly added to the scene. Defaults to True.
# transparent:	make the opacity of the actor depend on the scalar.
# vmax:	vmax is used to scale the colormap. If None, the max of the data will be used
# vmin:	vmin is used to scale the colormap. If None, the min of the data will be used
