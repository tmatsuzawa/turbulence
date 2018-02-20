import numpy as np
import h5py
import os
import sys
import matplotlib.pyplot as plt
import normal_mode_functions as nmf
import new_mode_functions as new_mf
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import scipy.ndimage as ndimage
import matplotlib.image as mpimg
import matplotlib.cm as cm

'''Plot the gyros' positions colored by their spinning speed, designed for high speed videos'''


def get_speed_and_plot(path, lowcutoff=50, minpower=0, clim=None, check=False):
    """Extract speeds from tracking data and plot as scatterplot with positions of the gyroscopes.

    Parameters
    ----------
    path : str
        path to hdf5 data with tracked positions

    Returns
    -------
    xy : npts x 2 float array
        positions of the gyroscopes with measured speeds
    speeds : npts x 1 float array
        speeds of each gyroscope
    """
    fn = os.path.join(path, 'com_data.hdf5')
    data = new_mf.load_linked_data_and_window(fn)
    coords = np.array([data[2], data[3]]).T

    tot_power, fft_list_x, fft_list_y, freq = nmf.ffts_and_add(data)

    maxes = []
    real_coords = []
    for i in xrange(len(fft_list_x)):
        new_fft = np.abs(fft_list_x[i][freq > lowcutoff]) ** 2 + np.abs(fft_list_y[i][freq > lowcutoff]) ** 2
        new_freq = freq[freq > lowcutoff]

        max_power_index = np.where(new_fft == np.max(new_fft))[0][0]
        print max_power_index

        if check:
            print 'new_freq[0:10] = ', new_freq[0:10]
            print 'new_fft[0:10] = ', new_fft[0:10]
            plt.plot(new_freq, new_fft)
            plt.plot([new_freq[max_power_index], new_freq[max_power_index]], [0, new_fft[max_power_index]], 'r-')
            plt.title('freq = {0:0.5f}'.format(new_freq[max_power_index]))
            plt.pause(1)
            plt.clf()

        if new_fft[max_power_index] > minpower:
            real_coords.append(coords[i])
            maxes.append(new_freq[max_power_index])

    if check:
        plt.clf()
        plt.close('all')

    real_coords = np.array(real_coords)
    maxes = np.array(maxes)
    patch = []

    for j in xrange(len(real_coords)):
        circ = Circle((real_coords[j, 0], real_coords[j, 1]), 30)
        patch.append(circ)
        print real_coords[j, 0], real_coords[j, 1]
        print 'maxes[jj] = ', maxes[j]
        print ' '

    fig = plt.figure()
    # img = mpimg.imread(path+ '/000_nb.png')
    # imgplot = plt.imshow(img, cmap=cm.Greys_r)
    p_ax = plt.gca()
    p = PatchCollection(patch, cmap='coolwarm', alpha=1.0)
    p.set_array(np.array(abs(maxes)))
    if clim is not None:
        p.set_clim(clim)
    p_ax.add_collection(p)
    plt.colorbar(p)

    plt.xlim(0, 800)
    plt.ylim(800, 0)

    plt.savefig(path + 'speed.png')
    plt.close()
    np.savetxt(path + 'speeds.txt', maxes)
    np.savetxt(path + 'coords.txt', real_coords)
    return coords, maxes


def get_speed_and_plot_withresolution(path, lowcutoff=50, hicutoff=325, minpower=0, clim=None, check=False):
    """Extract speeds from tracking data and plot as scatterplot with positions of the gyroscopes, including error bars
    fromresolution of fft

    Parameters
    ----------
    path : str
        path to hdf5 data with tracked positions

    Returns
    -------
    xy : npts x 2 float array
        positions of the gyroscopes with measured speeds
    speeds : npts x 1 float array
        speeds of each gyroscope
    resolution : float
        the bin size of the fft frequencies
    """
    fn = os.path.join(path, 'com_data.hdf5')
    data = new_mf.load_linked_data_and_window(fn)
    coords = np.array([data[2], data[3]]).T

    tot_power, fft_list_x, fft_list_y, freq = nmf.ffts_and_add(data, cutoff=0.)
    # plt.plot(freq, fft_list_x[1], 'o--')
    # plt.plot(freq, fft_list_y[1], 'o--')
    # # print 'fft_list_x = ', fft_list_x
    # # plt.plot(freq, tot_power, '.-')
    # plt.savefig(os.path.join(path, 'test.png'))

    maxes = []
    real_coords = []
    for i in xrange(len(fft_list_x)):
        keepinds = np.where(np.logical_and(freq > lowcutoff, freq < hicutoff))[0]
        new_fft = np.abs(fft_list_x[i][keepinds]) ** 2 + np.abs(fft_list_y[i][keepinds]) ** 2
        new_freq = freq[keepinds]
        # print 'keepinds = ', keepinds
        # print 'lowcutoff = ', lowcutoff
        # print 'hicutoff= ', hicutoff
        # print 'new_freq = ', new_freq
        # sys.exit()
        # print 'new_fft = ', new_fft
        # sys.exit()
        max_power_index = np.where(new_fft == np.max(new_fft))[0][0]

        if check:
            print 'new_freq[0:10] = ', new_freq[0:10]
            print 'new_fft[0:10] = ', new_fft[0:10]
            plt.semilogy(new_freq, new_fft)
            plt.plot([new_freq[max_power_index], new_freq[max_power_index]], [0, new_fft[max_power_index]], 'r-')
            plt.title('freq = {0:0.5f}'.format(new_freq[max_power_index]))
            plt.savefig(os.path.join(path, '{0:04d}'.format(i) + '.png'))
            plt.clf()

        if new_fft[max_power_index] > minpower:
            real_coords.append(coords[i])
            maxes.append(new_freq[max_power_index])

    if check:
        plt.clf()
        plt.close('all')

    real_coords = np.array(real_coords)
    maxes = np.array(maxes)
    patch = []

    for j in xrange(len(real_coords)):
        circ = Circle((real_coords[j, 0], real_coords[j, 1]), 30)
        patch.append(circ)
        print real_coords[j, 0], real_coords[j, 1]
        print 'maxes[jj] = ', maxes[j]
        print ' '

    fig = plt.figure()
    # img = mpimg.imread(path+ '/000_nb.png')
    # imgplot = plt.imshow(img, cmap=cm.Greys_r)
    p_ax = plt.gca()
    p = PatchCollection(patch, cmap='coolwarm', alpha=1.0)
    p.set_array(np.array(abs(maxes)))
    if clim is not None:
        p.set_clim(clim)
    p_ax.add_collection(p)
    plt.colorbar(p)

    plt.xlim(0, 800)
    plt.ylim(800, 0)

    plt.savefig(path + 'speed.png')
    plt.close()
    np.savetxt(path + 'speeds.txt', maxes)
    np.savetxt(path + 'coords.txt', real_coords)
    resolution = np.diff(new_freq)[0]
    np.savetxt(path + 'speeds_resolution.txt', np.array([resolution]))
    if check:
        print 'resolution = ', resolution
    return coords, maxes, resolution

if __name__ == "__main__":
    path = '/Volumes/labshared2/noah/20170430_test97gyro_withmotor_1p1inhoop_kagome/' \
           '20170430_t0_speedcheck_topmiddle_3000pps/'
    print path
    get_speed_and_plot(path)
