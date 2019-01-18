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


global rad2deg
rad2deg = 180. / np.pi

"""3D reconstruction of multilayer PIV
    Generate a 3d array from the pivlab outputs (hdf5) on each slice"""


parser = argparse.ArgumentParser('Sort PIVLab outputs of multilayer PIV experiments')
parser.add_argument('-dir', '--dir', help='Parent directory of the directories where pivlab outputs lie',
                    type=str,
                    default='/Volumes/bigraid/takumi/turbulence/3dprintedbox/multilayerPIV_Dp57mm_Do12p8mm/2018_11_04/PIV_W8_step2_data/')
parser.add_argument('-header', '--header', help='Header specifying PIVLab output directories. Default: PIV',
                    type=str, default='PIV')
parser.add_argument('-overwrite', '--overwrite', help='Default: False',
                    type=bool, default=False)
# parser.add_argument('-setup', '--setup', help='Header specifying PIVLab output directories. Default: PIV',
#                     type=str, default='PIV')
parser.add_argument('-delete', '--delete', nargs='+', help='z-indices to ignore to make an interpolated grid data',
                    type=int, default=[])
parser.add_argument('-mode', '--mode', help='interpolation method. choose from linear and nearest.',
                    type=str, default='nearest')


args = parser.parse_args()

datadirs = glob.glob(os.path.join(args.dir, args.header + '*'))

setup_files = glob.glob(os.path.join(args.dir, '*.3dsetup'))
ndata = len(datadirs)  # number of scans required to complete a scan of entire system

# Check if time_avg data exits. if so, perform interpolation.
avg_data_raw_path = args.dir + '/time_avg_field_raw.h5'
# avg_data_raw_path = args.dir + '/time_avg_field_raw_portion_1p0.h5'
# avg_data_raw_path = args.dir + '/time_avg_field_raw_portion_0p01.h5'


# plotting settings
vmin, vmax = 0, 3*10**4

if os.path.exists(avg_data_raw_path) and not args.overwrite:
    with h5py.File(avg_data_raw_path, 'r') as data:
        from scipy.interpolate import griddata
        x, y, z = np.asarray(data['x']), np.asarray(data['y']), np.asarray(data['z'])
        ux, uy, uz = np.asarray(data['ux_avg']), np.asarray(data['uy_avg']), np.asarray(data['uz_avg'])



        # width, height, depth = ux.shape

        # Read a sample setup file to extract scale
        # Load setup file
        setup_str = open(setup_files[0], 'rt').read()
        setup = {}
        dummy = {}
        exec ("from math import *", dummy)
        exec (setup_str, dummy, setup)

        data_spacing = float(setup['W']) / 2.
        scale, frame_rate = setup['scale'], setup['frame_rate']  # mm/px, frames per sec

        # conversion
        # x, y, z = x / data_spacing, y / data_spacing, z / data_spacing
        ux, uy, uz = ux * scale * frame_rate, uy * scale * frame_rate, uz * scale * frame_rate  # px/frame * mm/1px * frames/1sec

        # figure out order
        order = z[0, 0, :].argsort()
        data_list = [x, y, z, ux, uy, uz]
        for i in range(len(data_list)):
            data_list[i][:] = data_list[i][:, :, order]

        # Delete some some data values if you like
        def ignore_data_values(data, index_list=args.delete):
            return np.delete(data, index_list, axis=2)
        x, y, z, ux, uy, uz = map(ignore_data_values, (x, y, z, ux, uy, uz))


        # clean data
        mask_ux = process.get_mask_for_unphysical(ux, cutoffU=200., fill_value=99999., verbose=True)
        mask_uy = process.get_mask_for_unphysical(uy, cutoffU=200., fill_value=99999., verbose=True)
        mask_uz = process.get_mask_for_unphysical(uz, cutoffU=200., fill_value=99999., verbose=True)


        ux = process.interpolate_using_mask(ux, mask_ux)
        uy = process.interpolate_using_mask(uy, mask_uy)
        uz = process.interpolate_using_mask(uz, mask_uz)

        # filter
        ux = filters.gaussian_filter(ux, [0.5, 0.5, 0])
        uy = filters.gaussian_filter(uy, [0.5, 0.5, 0])
        uz = filters.gaussian_filter(uz, [0.5, 0.5, 0])


        x, y, z = x/data_spacing, y/data_spacing, z/data_spacing
        e = (ux**2 + uy**2)/2.


        for i in range(x.shape[2]):
            print i, np.min(z[..., i]), np.max(z[..., i]), np.mean(z[..., i])
            fig, ax, cc = graph.color_plot(x[..., i], y[..., i], e[..., i], cmap='plasma', vmin=vmin, vmax=vmax)
            graph.add_colorbar(cc, label=r'$\bar{E}_{2D}=\frac{1}{2}(\bar{U_x}^2)$', option='scientific')
            graph.labelaxes(ax, 'X (px)', 'Y (px)')
            graph.title(ax, '<z>=%.2f px' % np.mean(z[..., i]))
            fig.tight_layout()
            filename = '/time_avg_energy_raw_%s/zm%03d' % (args.mode, i)
            graph.save(args.dir + filename, ext='png', close=True, verbose=True)



        print x.shape, ux.shape
        xmin, xmax, ymin, ymax, zmin, zmax = np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z)

        points = zip(np.ravel(x), np.ravel(y), np.ravel(z)) # px after piv processing
        # values = np.ravel(ux)*scale*frame_rate #mm/s

        print 'make grid'

        # xx, yy, zz= np.mgrid[2:126:100j, 2:90:100j, -40:40:100j]
        xx, yy, zz= np.mgrid[xmin:xmax:200j, ymin:ymax:200j, zmin:zmax:200j]

        print '... Done'

        print 'make a griddata'
        grid_ux = griddata(points, np.ravel(ux), (xx, yy, zz), method=args.mode)
        grid_uy = griddata(points, np.ravel(uy), (xx, yy, zz), method=args.mode)
        # grid_uz = griddata(points, np.ravel(uz), (xx, yy, zz), method='nearest')
        grid_e = (grid_ux**2 + grid_uy**2)/2. # 2d energy

        for i in range(xx.shape[2]):
            fig, ax, cc = graph.color_plot(xx[..., i], yy[..., i], grid_e[..., i], cmap='plasma', vmin=vmin, vmax=vmax)
            graph.add_colorbar(cc, label=r'$\bar{E}_{2D}=\frac{1}{2}(\bar{U_x}^2)$', option='scientific')
            graph.labelaxes(ax, 'X (px)', 'Y (px)')
            fig.tight_layout()
            filename = '/time_avg_energy_%s/im%05d' % (args.mode, i)
            graph.save(args.dir + filename, ext='png', close=True)

        import library.tools.rw_data as rw
        savedata = {}
        savedata['x'] = xx
        savedata['y'] = yy
        savedata['ux'] = grid_ux
        savedata['uy'] = grid_uy
        savedata['energy'] = grid_e
        filepath = args.dir + '/grid_data_%s' % args.mode
        rw.write_hdf5_dict(filepath, savedata)
        # plt.show()


else:
    # Load setup data and figure out how to merge data
    laser_pos = np.empty(ndata)
    x0s = np.empty(ndata)
    zz2_top, zz2_bottom = np.empty(ndata), np.empty(ndata)
    z0s = np.empty(ndata)
    dthetas = np.empty(ndata)

    # make a list of dictionaries, Each dictionary has a different id
    dummy_func = lambda x: copy.deepcopy(x)
    setups = [dummy_func({}) for i in range(((ndata)))]

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
    data_spacing = setup['W'] / 2 # data spacing. number of pixels between neighboring data points. px
    # data_spacing = setup['W'] # data spacing. number of pixels between neighboring data points. px # fix for 11/5/18 data
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
        # theta = dtheta * ((setup['nslice'] - 1) / 2. - sliceno)
        theta = dtheta * (sliceno - (setup['nslice'] - 1) / 2.)
        dz = setup['z1'] / float(setup['nslice'] - 1) # mm
        # zoffset = dz * ((setup['nslice'] - 1) / 2. - sliceno) # mm
        zoffset = dz * (sliceno - (setup['nslice'] - 1) / 2.) # mm
        x_new = x # px
        y_new = y # px
        z_new = z0 + zoffset / scale + (x1 / scale + cine_width-x) * np.tan(theta / rad2deg)
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


    # Load hdf5 data
    for j, datadir in enumerate(datadirs):
        hdf5dir = os.path.join(datadir, 'hdf5data')
        h5files = glob.glob(hdf5dir + '/slice*.h5')


        # grab a setup file
        setup_id = int(datadir[-6])
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
            print sliceno,  setup['nslice'] - sliceno - 1
            sliceno = setup['nslice'] - sliceno -1
            print 'Check: Slice No must be ordered! 0-%d: %02d' % (setup['nslice'], sliceno)
            data = h5py.File(h5file, 'r')
            x_raw, y_raw = np.asarray(data['x']), np.asarray(data['y'])
            ux_raw, uy_raw = np.asarray(data['ux']), np.asarray(data['uy'])

            # position correction
            # uz is just a projection of ux_raw here
            x, y, z = transform_coord(x_raw, y_raw, z0 / scale, sliceno, setup) # 2d arrays
            ux, uy, uz = transform_vel(ux_raw, uy_raw , 0, dtheta * (sliceno - (setup['nslice']-1)/2.)) # 2d arrays

            # Insert data
            for k, data_raw in enumerate([x, y, z]):
                data_raw = np.swapaxes(data_raw, 0, 1) #(y,x)->(x,y)
                coord_data[k][...,  sliceno + j * setup['nslice']] = data_raw # (x,y,z)
            for k, data_raw in enumerate([ux, uy, uz]):
                data_raw = np.swapaxes(data_raw, 0, 1) #(y,x,t)->(x,y,t)
                vel_data[k][:,:, sliceno + j * setup['nslice'], :] = data_raw # (x,y,z,t)

            print 'j, sliceno, zmin, zmax, xmin, xmax: ', j, sliceno, np.min(z), np.max(z),  np.min(x), np.max(x)

            data.close()
            ## laser plane visualization
            # if i in [0, 8, 16]:
            #     fig = plt.figure()
            #     ax = fig.add_subplot(111, projection='3d')
            #     ax.scatter(x,y,z)
            #     ax.set_xlabel('X Label')
            #     ax.set_ylabel('Y Label')
            #     ax.set_zlabel('Z Label')
            #     plt.show()



    # from scipy.interpolate import griddata
    # data_spacing = float(data_spacing)
    # points = zip(np.ravel(coord_data[0]/data_spacing), np.ravel(coord_data[1]/data_spacing), np.ravel(coord_data[2])/data_spacing) # px after piv processing
    # values = zip(np.ravel(vel_data[0]*scale*frame_rate), np.ravel(vel_data[1]*scale*frame_rate), np.ravel(vel_data[2]*scale*frame_rate)) #mm/s
    # #
    # xmin, xmax, ymin, ymax, zmin, zmax = np.min(coord_data[0]), np.max(coord_data[0]),\
    #                                      np.min(coord_data[1]), np.max(coord_data[1]),\
    #                                      np.min(coord_data[2]), np.max(coord_data[2])
    # print 'len(points):', len(points), len(values)
    #
    # limits_raw = [xmin, xmax, ymin, ymax, zmin, zmax] # raw resolution
    # limits = [limit / data_spacing for limit in limits_raw]# resolution after piv processing
    # print limits_raw
    # print limits


    # Time-average fields
    ux_avg, uy_avg, uz_avg = np.zeros_like(newshape1), np.zeros_like(newshape1), np.zeros_like(newshape1)
    ux2_avg, uy2_avg, uz2_avg = np.zeros_like(newshape1), np.zeros_like(newshape1), np.zeros_like(newshape1)
    avg_data = [ux_avg, uy_avg, uz_avg]
    avg2_data = [ux2_avg, uy2_avg, uz2_avg]
    print vel_data[0].shape
    print 'Time-averaging...'
    # 60000 frames in total (i.e. 30 sec.)
    # 60000 * 2/3 * duty cycle = 34000 frames
    # 2000 frames for each slice. interal: 60 frames = 30 ms
    # used a pair of images to extract a v field -> 1000 data points
    ratio = 1.0
    duration = vel_data[0].shape[3]
    for i in range(3):
        print '%d / 3' % i
        avg_data[i] = np.nanmean(vel_data[i][..., 0:int(duration*ratio)], axis=3)
        avg2_data[i] = np.nanmean(vel_data[i][..., 0:int(duration*ratio)]**2, axis=3)

    import library.tools.rw_data as rw
    griddata_path = args.dir + '/time_avg_field_raw_portion_%s' % (str(ratio).replace('.', 'p'))
    data = {}
    data['ux_avg'] = avg_data[0]
    data['uy_avg'] = avg_data[1]
    data['uz_avg'] = avg_data[2]  # bogus
    data['ux2_avg'] = avg2_data[0]
    data['uy2_avg'] = avg2_data[1]
    data['uz2_avg'] = avg2_data[2]  # bogus
    data['x'] = coord_data[0]
    data['y'] = coord_data[1]
    data['z'] = coord_data[2]
    rw.write_hdf5_dict(griddata_path, data)


    # visualize laser sheet position
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    nn = 20
    colors = graph.get_first_n_colors_from_color_cycle(ndata)
    for i in range(ndata):
        # ax.scatter(coord_data[0][::nn, ::nn, i * 17: (i + 1) * 17], coord_data[1][::nn, ::nn, i * 17: (i + 1) * 17],
        #             coord_data[2][::nn, ::nn, i * 17: (i + 1) * 17],
        #             alpha=0.3)
        ax.scatter(coord_data[0][::nn, ::nn, i * 17: (i * 17)+1], coord_data[1][::nn, ::nn, i * 17:  (i * 17)+1],
                    coord_data[2][::nn, ::nn, i * 17: (i * 17)+1],
                    alpha=0.3, color=colors[i])
        ax.scatter(coord_data[0][::nn, ::nn, (i * 17) + 15: (i * 17) + 16], coord_data[1][::nn, ::nn, (i * 17) + 15:  (i * 17) + 16],
                   coord_data[2][::nn, ::nn, (i * 17) + 15: (i * 17) + 16],
                   alpha=0.3, marker='x',s=60, color=colors[i])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.view_init(azim=270, elev=00)
    fig.tight_layout()
    graph.save(args.dir + '/laser_pos2', ext='png')

    plt.show()





    # coord_shape, vel_shape = coord_data[0].shape, vel_data[0].shape
    # z_a, z_b = -50, 50
    # ind1 = (coord_data[2]/data_spacing) > z_a
    # ind2 = (coord_data[2]/data_spacing) < z_b
    # coord_ind = np.reshape(ind1 * ind2, coord_shape)
    # vel_ind = coord_ind.reshape(coord_shape[0], coord_shape[1], coord_shape[2], vel_shape[3])
    # print np.sum(coord_ind), coord_ind.shape, vel_ind.shape, vel_data[0][vel_ind].shape
    #
    # x_t, y_t, z_t = coord_data[0][coord_ind]/data_spacing, coord_data[1][coord_ind]/data_spacing, coord_data[2][coord_ind]/data_spacing
    # ux_t, uy_t, uz_t = vel_data[0][vel_ind]*scale*frame_rate, vel_data[1][vel_ind]*scale*frame_rate, vel_data[2][vel_ind]*scale*frame_rate
    #
    #
    #
    #
    # print 'make grid'
    #
    # # xx, yy, zz= np.mgrid[2:126:100j, 2:90:100j, -40:40:100j]
    # xx, yy, zz= np.mgrid[xmin:xmax:100j, ymin:ymax:100j, zmin:zmax:100j]
    #
    # points = zip(x_t, y_t, z_t)
    # values = ux_t
    # print '... Done'
    # print vel_data[0].shape
    #
    # print 'make a griddata'
    # # grid_ux = griddata(points, ux_t, (xx, yy, zz), method='nearest')
    # # grid_uy = griddata(points, uy_t, (xx, yy, zz), method='nearest')
    # # grid_uz = griddata(points, uz_t, (xx, yy, zz), method='nearest')
    # grid_ux = griddata(points, ux_t, (xx, yy, zz), method='linear')
    # grid_uy = griddata(points, uy_t, (xx, yy, zz), method='linear')
    # grid_uz = griddata(points, uz_t, (xx, yy, zz), method='linear')
    #
    #
    #
    # # print grid_ux.shape
    # # plt.pcolormesh(xx[..., 0], yy[..., 0], grid_ux[..., 5, 0])
    # # plt.colorbar()
    # # plt.show()
    #
    # import library.tools.rw_data as rw
    # griddata_path = args.dir + '/3dinterpolated_data_200x200x200_center_linear'
    # data = {}
    # data['ux'] = grid_ux
    # data['uy'] = grid_uy
    # data['uz'] = grid_uz  # bogus
    # data['x'] = xx
    # data['y'] = yy
    # data['z'] = zz
    # rw.write_hdf5_dict(griddata_path, data)
