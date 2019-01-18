import scipy
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
import library.basics.formatstring as fs
import library.display.graph as graph
import turbulence.manager.pivlab2hdf5 as pivlab2hdf5
import os
from math import *
import sys
import h5py
from mpl_toolkits.mplot3d import Axes3D

global rad2deg
rad2deg = 180. / np.pi

"""3D reconstruction of multilayer PIV
    Generate a 3d array from the pivlab outputs (hdf5) on each slice"""


parser = argparse.ArgumentParser('Sort PIVLab outputs of multilayer PIV experiments')
parser.add_argument('-dir', '--dir', help='Parent directory of the directories where pivlab outputs lie',
                    type=str,
                    default='/Volumes/bigraid/takumi/turbulence/3dprintedbox/multilayerPIV_Dp57mm_Do12p8mm/2018_11_04/PIV_W8_step2_data/sample/')
parser.add_argument('-header', '--header', help='Header specifying PIVLab output directories. Default: PIV',
                    type=str, default='PIV')
# parser.add_argument('-setup', '--setup', help='Header specifying PIVLab output directories. Default: PIV',
#                     type=str, default='PIV')

args = parser.parse_args()

datadirs = glob.glob(os.path.join(args.dir, args.header + '*'))
setup_files = glob.glob(os.path.join(args.dir, '*.3dsetup'))
ndata = len(datadirs)  # number of scans required to complete a scan of entire system

# Load setup data and figure out how to merge data
laser_pos = np.empty(ndata)
x0s = np.empty(ndata)
zz2_top, zz2_bottom = np.empty(ndata), np.empty(ndata)
z0s = np.empty(ndata)
dthetas = np.empty(ndata)
setups = [{}, {}, {}]

for i in range(ndata):
    # Load setup file
    setup_str = open(setup_files[i], 'rt').read()
    setup = setups[i]
    dummy = {}
    exec("from math import *", dummy)
    exec (setup_str, dummy, setup)
    # print setup
    laser_pos[i] = setup['z0']
    x0s[i] = setup['z1'] / 2. / np.arctan(setup['theta'] / rad2deg)
    z0s[i] = setup['z0']
    zz2_top[i] = setup['z0'] + (x0s[i] + setup['x2']) * np.tan(setup['theta'] / rad2deg)
    zz2_bottom[i] = setup['z0'] - (x0s[i] + setup['x2']) * np.tan(setup['theta'] / rad2deg)
    dthetas[i] = setup['theta'] / (setup['nslice'] - 1)
    setup['id'] = int(setup_files[i][-9:-8]) # this id corresponds to the location of laser
    frame_rate = float(setup['frame_rate'])
scale_raw = setup['scale'] # mm/px
# data_spacing = setup['W'] / 2 # data spacing. number of pixels between neighboring data points. px
data_spacing = setup['W'] # data spacing. number of pixels between neighboring data points. px # fix for 11/5/18 data
depth_mm = np.max(zz2_top) - np.min(zz2_bottom) # mm

depth = int(depth_mm / scale_raw / data_spacing) # data point
width, height = int(setup['cine_width'] / data_spacing), int(setup['cine_height'] / data_spacing)  # data point, data point

print width, height,  depth
# initialize 3d array
# data = np.zeros((width, height, depth))




def transform_coord(x, y, z0, sliceno, setup):
    """

    Parameters
    ----------
    x: x coordinate in pivlab outputs  (px)
    y: y coordinate in pivlab outputs  (px)
    z0: z postion of the central plane  (px)
    theta: angle of the plane from the central plane in DEGREES
        theta > 0 for sliceno = 0-7
        theta < 0 for sliceno = 9-16

    Returns
    -------
    x_new, y_new

    """
    x1 = setup['x1'] # mm
    cine_width = setup['cine_width'] # mm
    scale = setup['scale'] # mm/px
    dtheta = setup['theta'] / float(setup['nslice'] - 1)
    theta = dtheta * ((setup['nslice'] - 1) / 2. - sliceno)
    dz = setup['z1'] / float(setup['nslice'] - 1) # mm
    zoffset = dz * ((setup['nslice'] - 1) / 2. - sliceno) # mm
    x_new = x # px
    y_new = y # px
    print x1 / scale, cine_width, theta, np.tan(theta / rad2deg)
    z_new = z0 - zoffset / scale - (x1 / scale + cine_width-x) * np.tan(theta / rad2deg)
    # print 'x1, cinewidth, theta', x1, cine_width, theta
    return x_new, y_new, z_new


def transform_vel(ux, uy, uz0, theta):
    """
    Parameters
    ----------
    ux: ux in pivlab outputs  (px/frame)
    uy: uy in pivlab outputs  (px/frame)
    uz0: guess of rms uz on the central plane  (px/frame)... Should be zero except test purposes
    theta: angle of the plane from the central plane in DEGREES

    Returns
    -------
    x_new, y_new

    """
    ux_new = ux * np.cos(-theta / rad2deg)
    uy_new = uy
    uz_new = uz0 + ux * np.sin(-theta / rad2deg)
    return ux_new, uy_new, uz_new


def get_setup(setups, id):
    for setup in setups:
        if setup['id']==id:
            return setup


def draw_fov_for_lens_with_unfixed_f(w_fov_min, h_fov_min, w_fov_max, h_fov_max):
    """

    Parameters
    ----------
    w_fov_min
    h_fov_min
    w_fov_max
    h_fov_max

    Returns
    -------

    """

# Load hdf5 data
for j, datadir in enumerate(datadirs):
    hdf5dir = os.path.join(datadir, 'hdf5data')
    h5files = glob.glob(hdf5dir + '/slice*.h5')
    # grab a setup file
    setup_id = int(datadir[-1])
    setup = get_setup(setups, setup_id)
    print setup
    z0, scale = setup['z0'],  setup['scale']  # mm, mm/px
    dtheta = setup['theta'] / float(setup['nslice'] - 1)

    if j == 0:
        # get velocity field data size
        with h5py.File(h5files[0],'r') as data:
            rows, cols, duration = data['ux'].shape # (y, x, t)

        # initialize arrays to store experimental data
        newshape1 = (cols, rows, setup['nslice']*ndata) # used for coordinate data organizing
        newshape2 = (cols, rows, setup['nslice']*ndata, duration) # used for coordinate data organizing
        xdata, ydata, zdata = np.empty(newshape1), np.empty(newshape1), np.empty(newshape1)
        uxdata, uydata, uzdata = np.empty(newshape2), np.empty(newshape2), np.empty(newshape2)
        coord_data, vel_data = [xdata, ydata, zdata], [uxdata, uydata, uzdata]

    for i, h5file in enumerate(sorted(h5files)):
        sliceno = int(fs.get_float_from_str(h5file, 'slice', '.h5'))
        print sliceno,  setup['nslice'] - sliceno -1
        # sliceno = setup['nslice'] - sliceno -1
        print 'Check: Slice No must be ordered! 0-%d: %02d' % (setup['nslice'], sliceno)

        data = h5py.File(h5file, 'r')
        x_raw, y_raw = np.asarray(data['x']), np.asarray(data['y'])
        ux_raw, uy_raw = np.asarray(data['ux']), np.asarray(data['uy'])

        # position correction
        # uz is just a projection of ux_raw here
        x, y, z = transform_coord(x_raw, y_raw, z0 / scale, sliceno, setup) # 2d arrays
        ux, uy, uz = transform_vel(ux_raw, uy_raw , 0, dtheta * ((setup['nslice']-1)/2.-sliceno)) # 2d arrays

        # Insert data
        for k, data_raw in enumerate([x, y, z]):
            data_raw = np.swapaxes(data_raw, 0, 1) #(y,x)->(x,y)
            coord_data[k][...,  sliceno + j * setup['nslice']] = data_raw # (x,y,z)
        for k, data_raw in enumerate([ux, uy, uz]):
            data_raw = np.swapaxes(data_raw, 0, 1) #(y,x,t)->(x,y,t)
            vel_data[k][:,:, sliceno + j * setup['nslice'], :] = data_raw # (x,y,z,t)

        print 'j, sliceno, zmin, zmax, xmin, xmax: ', j, sliceno, np.min(z), np.max(z),  np.min(x), np.max(x)


        # fig1, ax1, cc1 = graph.color_plot(x_raw, y_raw, ux_raw[..., 0], fignum=1, subplot=131, cmap='RdBu', vmin=-0.7, vmax=0.7)
        # fig1, ax2, cc2 = graph.color_plot(x, y, ux[..., 0], fignum=1, subplot=132, cmap='RdBu', vmin=-0.7, vmax=0.7)
        # fig1, ax3, cc3 = graph.color_plot(x, y, ux[..., 0]-ux_raw[..., 0], fignum=1, subplot=133, cmap='RdBu',
        #                                   figsize=(24, 10), vmin=-0.0002, vmax=0.0002)
        # graph.add_colorbar(cc1, ax=ax1)
        # graph.add_colorbar(cc2, ax=ax2)
        # graph.add_colorbar(cc3, ax=ax3)
        # graph.suptitle('z=%.2f' % np.min(z))
        # graph.show()

        # try:
        #     #
        #     newshape = (rows, cols, 1) # (y, x, z)
        #     print '-----------'
        #     print x.reshape(newshape).shape
        #     x = np.swapaxes(x.reshape(newshape), 0, 1) # (x, y, z)
        #
        #
        #     print x.shape, xdata.shape
        #     xdata = np.stack((xdata, x), axis=2) # 3d array for x, y, z  (x,y,z)
        #     print xdata.shape
        #     # uxdata = np.dstack((uxdata, ux.reshape(newshape)))  # 4d array for ux, uy, uz (z,y,x, t)
        # except NameError:
        #     newshape = (rows, cols, 1)
        #     xdata = x.reshape((rows, cols, 1)) # (y,x,z)
        #     # print xdata.shape
        #     xdata = np.swapaxes(xdata, 0, 1)# (x,y,z)
        #     print xdata.shape
        #
        data.close()
        # if i in [0, 8, 16]:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111, projection='3d')
        #     nn = 20
        #     ax.scatter(x,y,z)
        #     ax.set_xlabel('X Label')
        #     ax.set_ylabel('Y Label')
        #     ax.set_zlabel('Z Label')
        #
        #     plt.show()



from scipy.interpolate import griddata
data_spacing = float(data_spacing)
points = zip(np.ravel(coord_data[0]/data_spacing), np.ravel(coord_data[1]/data_spacing), np.ravel(coord_data[2])/data_spacing) # px after piv processing
values = zip(np.ravel(vel_data[0]*scale*frame_rate), np.ravel(vel_data[1]*scale*frame_rate), np.ravel(vel_data[2]*scale*frame_rate)) #mm/s
#
xmin, xmax, ymin, ymax, zmin, zmax = np.min(coord_data[0]), np.max(coord_data[0]),\
                                     np.min(coord_data[1]), np.max(coord_data[1]),\
                                     np.min(coord_data[2]), np.max(coord_data[2])
print 'len(points):', len(points), len(values)

limits_raw = [xmin, xmax, ymin, ymax, zmin, zmax] # raw resolution
limits = [limit / data_spacing for limit in limits_raw]# resolution after piv processing
print limits_raw
print limits

coord_shape, vel_shape = coord_data[0].shape, vel_data[0].shape
z_a, z_b = -50, 50
ind1 = (coord_data[2]/data_spacing) > z_a
ind2 = (coord_data[2]/data_spacing) < z_b
coord_ind = np.reshape(ind1 * ind2, coord_shape)
vel_ind = coord_ind.reshape(coord_shape[0], coord_shape[1], coord_shape[2], vel_shape[3])
print np.sum(coord_ind), coord_ind.shape, vel_ind.shape, vel_data[0][vel_ind].shape

x_t, y_t, z_t = coord_data[0][coord_ind]/data_spacing, coord_data[1][coord_ind]/data_spacing, coord_data[2][coord_ind]/data_spacing
ux_t, uy_t, uz_t = vel_data[0][vel_ind]*scale*frame_rate, vel_data[1][vel_ind]*scale*frame_rate, vel_data[2][vel_ind]*scale*frame_rate



#
# print 'make grid'
#
# # xx, yy, zz= np.mgrid[2:126:100j, 2:90:100j, -40:40:100j]
# xx, yy, zz= np.mgrid[xmin:xmax:200j, ymin:ymax:200j, zmin:zmax:200j]
#
# points = zip(x_t, y_t, z_t)
# values = ux_t
# print '... Done'
# print vel_data[0].shape
#
# print 'make a griddata'
# grid_ux = griddata(points, ux_t, (xx, yy, zz), method='nearest')
# grid_uy = griddata(points, uy_t, (xx, yy, zz), method='nearest')
# grid_uz = griddata(points, uz_t, (xx, yy, zz), method='nearest')
# # grid_ux = griddata(points, ux_t, (xx, yy, zz), method='linear')
# # grid_uy = griddata(points, uy_t, (xx, yy, zz), method='linear')
# # grid_uz = griddata(points, uz_t, (xx, yy, zz), method='linear')
#
#
#
# # print grid_ux.shape
# # plt.pcolormesh(xx[..., 0], yy[..., 0], grid_ux[..., 5, 0])
# # plt.colorbar()
# # plt.show()
#
# import library.tools.rw_data as rw
# griddata_path = args.dir + '/3dinterpolated_data_200x200x200_center_nearest'
# data = {}
# data['ux'] = grid_ux
# data['uy'] = grid_uy
# data['uz'] = grid_uz  # bogus
# data['x'] = xx
# data['y'] = yy
# data['z'] = zz
# rw.write_hdf5_dict(griddata_path, data)



# print len(points[::100])
# # visualize space where the laser swept
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
nn = 20
for i in range(ndata):
    ax.scatter(coord_data[0][::nn,::nn, i * 17: (i+1) * 17], coord_data[1][::nn,::nn, i * 17: (i+1) * 17], coord_data[2][::nn,::nn, i * 17: (i+1) * 17],
               alpha=0.3)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.view_init(azim=270, elev=00)
fig.tight_layout()
# graph.save(args.dir + '/laser_pos2', ext='png')

plt.show()
#

resultdir = '/Volumes/bigraid/takumi/turbulence/3dprintedbox/multilayerPIV_Dp57mm_Do12p8mm/2018_11_04/PIV_W8_step2_data/sample/raw_results/'
zzz = np.arange(51)

for i in range(len(zzz)):
    xx, yy = coord_data[0][:,:, i], coord_data[1][:,:, i]
    ux = vel_data[0][:,:,i, 0]
    plt.figure()
    plt.pcolormesh(xx, yy, ux, cmap='RdBu', vmin=-0.8, vmax=0.8)
    plt.colorbar()
    graph.save(resultdir + 'z%d' % i, ext='png')
    plt.close()

# print 750*scale