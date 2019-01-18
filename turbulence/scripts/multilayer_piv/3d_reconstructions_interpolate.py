import scipy
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
import library.basics.formatstring as fs
import library.display.graph as graph
import turbulence.manager.pivlab2hdf5 as pivlab2hdf5
import library.tools.process_data as process
import os
from math import *
import sys
import h5py
from mpl_toolkits.mplot3d import Axes3D
import copy
import scipy.ndimage.filters as filters

from scipy.spatial import KDTree

global rad2deg
rad2deg = 180. / np.pi

"""3D reconstruction of multilayer PIV
    Generate a 3d array from the pivlab outputs (hdf5) on each slice"""


parser = argparse.ArgumentParser('Sort PIVLab outputs of multilayer PIV experiments')
parser.add_argument('-f', '--f', help='RAW time-averaged h5 file',
                    type=str,
                    default='/Volumes/bigraid/takumi/turbulence/3dprintedbox/multilayerPIV_Dp120mm_Do25p6mm/2018_12_11/PIV_W16_step2_data/time_avg_field_raw_portion_1p0.h5')

args = parser.parse_args()

dir = os.path.split(args.f)[0]
setup_files = glob.glob(os.path.join(dir, '*.3dsetup'))

with h5py.File(args.f, 'r') as data:
    print data.keys()
    x, y, z = np.asarray(data['x']), np.asarray(data['y']), np.asarray(data['z'])
    ux, uy, uz = np.asarray(data['ux_avg']), np.asarray(data['uy_avg']), np.asarray(data['uz_avg'])
    energy = (ux ** 2 + uy ** 2 + uz ** 2) / 2.

    # manually correct data
    z1, z2 = 0, 25
    deltaz = 5. # mm
    z[..., z1:z2] += deltaz

    graph.color_plot(x[..., 0], y[..., 0], energy[..., 90], vmin=0, vmax=0.4)
    graph.show()
    z1, z2 = 85, 102
    deltaz = -30. # mm
    z[..., z1:z2] += deltaz

    pts = np.asarray(zip(x.ravel(), y.ravel(), z.ravel()))
    energy_ravelled = energy.ravel() #np array

    # Make a KD tree
    # print len(data), len(pts), pts[0:10]
    print '... making a kd tree'
    tree = KDTree(pts, leafsize=10)
    print '... done'

    # Make coordinates
    xmin, xmax, ymin, ymax, zmin, zmax = np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)
    deltax = (x[1, 0, 0] - x[0, 0, 0]) / 2. * np.sqrt(3)
    dx = x[1, 0, 0] - x[0, 0, 0]



    nx, ny, nz = int((xmax-xmin)/dx), int((ymax-ymin)/dx), int((ymax-ymin)/dx)
    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    zi = np.linspace(zmin, zmax, nz)
    yyi, xxi, zzi = np.meshgrid(xi, yi, zi)  # y, x, z

    print nx, ny, nz, xxi.shape

    # Prepare an empty grid for interpolated values
    eei = np.empty(xxi.shape)

    for i in range(nx):
        print i, ' / ', nx, '  ', ny*nz
        for j in range(ny):
            for k in range(nz):
                pos = (xi[i], yi[j], zi[k])

                indices = tree.query_ball_point(pos, deltax)

                if indices == []:
                    dist, indices = tree.query(pos, k=1) # get the nearest neighbor at least
                    print '... no points found in the radius'
                    print dist, pos, pts[indices]
                # print pos, deltax, pts[indices], i, j, k

                eei[j][i][k] = np.nanmean(energy_ravelled[indices])
                # print indices, eei[i][j][k]

                # break
                # distances, indices = tree.query(pos, k=3) # get 2 nearest neighbors
                # print distances


    import library.tools.rw_data as rw
    savedata = {}
    savedata['x'] = xxi
    savedata['y'] = yyi
    savedata['z'] = zzi
    savedata['energy'] = eei
    rw.write_hdf5_dict(dir + '/interp_data_1p0_test_deltaz_%.1f_-30' % deltaz, savedata)


    ## Does not work.
    # from scipy.interpolate import LinearNDInterpolator
    # int_func = LinearNDInterpolator(pts, data)
    ## griddata does not work

        #
        # # width, height, depth = ux.shape
        #
        # # Read a sample setup file to extract scale
        # # Load setup file
        # setup_str = open(setup_files[0], 'rt').read()
        # setup = {}
        # dummy = {}
        # exec ("from math import *", dummy)
        # exec (setup_str, dummy, setup)
        #
        # data_spacing = float(setup['W']) / 2.
        # scale, frame_rate = setup['scale'], setup['frame_rate']  # mm/px, frames per sec
        #
        # # conversion
        # # x, y, z = x / data_spacing, y / data_spacing, z / data_spacing
        # ux, uy, uz = ux * scale * frame_rate, uy * scale * frame_rate, uz * scale * frame_rate  # px/frame * mm/1px * frames/1sec
        #
        # # figure out order
        # order = z[0, 0, :].argsort()
        # data_list = [x, y, z, ux, uy, uz]
        # for i in range(len(data_list)):
        #     data_list[i][:] = data_list[i][:, :, order]
        #
        # # Delete some some data values if you like
        # def ignore_data_values(data, index_list=args.delete):
        #     return np.delete(data, index_list, axis=2)
        # x, y, z, ux, uy, uz = map(ignore_data_values, (x, y, z, ux, uy, uz))
        #
        #
        # # clean data
        # mask_ux = process.get_mask_for_unphysical(ux, cutoffU=200., fill_value=99999., verbose=True)
        # mask_uy = process.get_mask_for_unphysical(uy, cutoffU=200., fill_value=99999., verbose=True)
        # mask_uz = process.get_mask_for_unphysical(uz, cutoffU=200., fill_value=99999., verbose=True)
        #
        #
        # ux = process.interpolate_using_mask(ux, mask_ux)
        # uy = process.interpolate_using_mask(uy, mask_uy)
        # uz = process.interpolate_using_mask(uz, mask_uz)
        #
        # # filter
        # ux = filters.gaussian_filter(ux, [0.5, 0.5, 0])
        # uy = filters.gaussian_filter(uy, [0.5, 0.5, 0])
        # uz = filters.gaussian_filter(uz, [0.5, 0.5, 0])
        #
        #
        # x, y, z = x/data_spacing, y/data_spacing, z/data_spacing
        # e = (ux**2 + uy**2)/2.
        #
        #
        # for i in range(x.shape[2]):
        #     print i, np.min(z[..., i]), np.max(z[..., i]), np.mean(z[..., i])
        #     fig, ax, cc = graph.color_plot(x[..., i], y[..., i], e[..., i], cmap='plasma', vmin=vmin, vmax=vmax)
        #     graph.add_colorbar(cc, label=r'$\bar{E}_{2D}=\frac{1}{2}(\bar{U_x}^2)$', option='scientific')
        #     graph.labelaxes(ax, 'X (px)', 'Y (px)')
        #     graph.title(ax, '<z>=%.2f px' % np.mean(z[..., i]))
        #     fig.tight_layout()
        #     filename = '/time_avg_energy_raw_%s/zm%03d' % (args.mode, i)
        #     graph.save(args.dir + filename, ext='png', close=True, verbose=True)
        #
        #
        #
        # print x.shape, ux.shape
        # xmin, xmax, ymin, ymax, zmin, zmax = np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)
        #
        # points = zip(np.ravel(x), np.ravel(y), np.ravel(z)) # px after piv processing
        # # values = np.ravel(ux)*scale*frame_rate #mm/s
        #
        # print 'make grid'
        #
        # # xx, yy, zz= np.mgrid[2:126:100j, 2:90:100j, -40:40:100j]
        # xx, yy, zz= np.mgrid[xmin:xmax:200j, ymin:ymax:200j, zmin:zmax:200j]
        #
        # print '... Done'
        #
        # print 'make a griddata'
        # grid_ux = griddata(points, np.ravel(ux), (xx, yy, zz), method=args.mode)
        # grid_uy = griddata(points, np.ravel(uy), (xx, yy, zz), method=args.mode)
        # # grid_uz = griddata(points, np.ravel(uz), (xx, yy, zz), method='nearest')
        # grid_e = (grid_ux**2 + grid_uy**2)/2. # 2d energy
        #
        # for i in range(xx.shape[2]):
        #     fig, ax, cc = graph.color_plot(xx[..., i], yy[..., i], grid_e[..., i], cmap='plasma', vmin=vmin, vmax=vmax)
        #     graph.add_colorbar(cc, label=r'$\bar{E}_{2D}=\frac{1}{2}(\bar{U_x}^2)$', option='scientific')
        #     graph.labelaxes(ax, 'X (px)', 'Y (px)')
        #     fig.tight_layout()
        #     filename = '/time_avg_energy_%s/im%05d' % (args.mode, i)
        #     graph.save(args.dir + filename, ext='png', close=True)
        #
        # import library.tools.rw_data as rw
        # savedata = {}
        # savedata['x'] = xx
        # savedata['y'] = yy
        # savedata['ux'] = grid_ux
        # savedata['uy'] = grid_uy
        # savedata['energy'] = grid_e
        # filepath = args.dir + '/grid_data_%s' % args.mode
        # rw.write_hdf5_dict(filepath, savedata)
        # # plt.show()
