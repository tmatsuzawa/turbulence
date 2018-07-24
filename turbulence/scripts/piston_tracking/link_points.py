import glob
import h5py
import matplotlib.pyplot as plt
import movie_instance as mi
import numpy as np
import os
import pandas as pd
import seaborn
import sys
import time
import tracking_helper_functions as thf
from settings import tracking_settings
from scipy.signal import savgol_filter
try:
    from video_analyzing.new_mode_functions import load_linked_data_and_window
except:
    from new_mode_functions_reserve import load_linked_data_and_window
import sys

'''description'''


def link_points(root_dir, skip_savepath=False, verbose=False, cutoff_distance=None):
    """

    Parameters
    ----------
    root_dir
    skip_savepath
    verbose

    Returns
    -------

    """
    file = os.path.join(root_dir, 'steps/steps.hdf5')
    data = h5py.File(file, 'r')
    keys = data.keys()

    if verbose:
        print 'link_points.py: keys = ', keys

    for ii in xrange(len(keys)):
        key = keys[ii]
        dat = len(np.array(data[key]))
        if dat == 54:
            start_key = key
            break
        else:
            start_key = keys[0]

    compare = np.array(data[start_key])

    if cutoff_distance is None and np.shape(compare)[0] == 1:
        # there is only one particle, so make cutoff distance large
        cutoff_distance = 1e10
    elif cutoff_distance is None:
        # Measure interparticle distance to get cutoff distance for which a particle is linked between two timesteps
        # Grab point in the middle of the lattice
        xytmp = compare[:, 1:3]
        midind = np.argmin(np.sum(np.abs(xytmp - np.mean(xytmp, axis=0)), axis=1))
        nbrs = np.setdiff1d(np.arange(len(xytmp)), np.array([midind]))
        diffdist = xytmp[midind] - xytmp[nbrs]
        dists = diffdist[:, 0]**2 + diffdist[:, 1]**2
        cutoff_distance = np.sqrt(np.min(dists)) * 0.7
        print 'link_points: cutoff_distance = ', cutoff_distance

    fig = plt.figure()
    plt.scatter(compare[:, 1], compare[:, 2], c=range(len(compare)), cmap=plt.cm.coolwarm)

    # label by index
    if len(compare) > 1:
        # There are multiple particles. plot them by number
        ii = 0
        diffx = np.diff(compare[:, 1])
        print 'link_points: diffx = ', diffx
        # define space to separate index labels from particles
        # space = np.min(diffx[np.abs(diffx > 5)]) * 0.2
        space = np.min(diffx) * 0.2
        for xypos in compare:
            plt.text(xypos[1] + space, xypos[2], str(ii))
            ii += 1
        plt.gca().set_aspect(1)
        plt.savefig(os.path.join(root_dir, 'color_by_number.png'))

    # Create hdf5 file to dump xy data into
    path = os.path.join(root_dir, 'com_data.hdf5')
    new = h5py.File(path, "w")

    for ii in xrange(len(compare)):
        if ii % 5 == 0:
            print 'link_points.py: particle ', ii, ' / ', len(compare)
        single_data = []
        times = []
        count = 0
        pt = compare[ii]
        # for each timestep, there is a key in keys
        for key in keys:
            step_data = np.array(data[key])
            t = step_data[0, 0]
            dist = np.sum(((step_data - pt) ** 2)[:, 1:], axis=1)

            ind = np.where(dist < cutoff_distance)[0]
            if len(ind) > 0:
                times.append(t)
                count += 1
                single_data.append(step_data[ind][0, 1:])
                pt = step_data[ind][0]
            else:
                times.append(t)
                count += 1
                single_data.append(pt[1:])

        single_data = np.array(single_data)

        single_data[:, 0] = savgol_filter(single_data[:, 0], 11, 1)
        single_data[:, 1] = savgol_filter(single_data[:, 1], 11, 1)
        key_name = '%03d' % ii

        dset = new.create_dataset(key_name, np.shape(single_data), dtype='float', data=single_data)

        image_path = os.path.join(root_dir, 'piston_path_images/')
        if not os.path.exists(image_path):
            os.mkdir(image_path)

        image_path = os.path.join(image_path, '%03d.png' % ii)
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        len_single = len(single_data)
        if not skip_savepath:
            ax1.plot(times[:len_single], single_data[:, 0])

            # make sign for where the particle is in space
            ax1.set_title(r'$x$ displacement for particle ' + str(ii) + ' at ({0:0.0f}'.format(single_data[0, 0]) +
                          ',{0:0.0f}'.format(single_data[0, 1]) + ')')
            ax2.plot(compare[:, 1], compare[:, 2], 'k.')
            ax2.plot(compare[ii, 1], compare[ii, 2], 'ro')
            ax2.axis('scaled')
            ax2.axis('off')
            ax1.set_xlabel('time [s]')
            ax1.set_ylabel(r'$x$ displacement')
            plt.savefig(image_path)
            plt.close()

    dset = new.create_dataset('times', np.shape(times), dtype='float', data=times)
    new.close()
    data.close()


def filter_by_frequency(data_path, frequency):
    """

    Parameters
    ----------
    data_path
    frequency

    Returns
    -------

    """
    data = h5py.File(data_path, 'r')
    keys = data.keys()

    # for key in keys:


if __name__ == '__main__':
    root_dir = '/Volumes/labshared2/Lisa/2017_02_21/tracked/7p_0p0A_5p5A_1_2/'

    path = '/Volumes/GetIt/saved_stuff/2017_05_18/1p77_1_2017_05_18/'
    # [np.array(x), np.array(y), np.array(x_mean), np.array(y_mean), np.array(time)]

    x, y, x_mean, y_mean, time = load_linked_data_and_window(path + 'com_data.hdf5', window=False)

    fft_freq = np.fft.fftfreq(len(x[0]), time[1] - time[0])

    diff = np.abs(1.77 - fft_freq)
    closest = np.where(diff == np.min(diff))[0][0]
    closest_freq = fft_freq[closest]
    print 'link_points.py: closest_freq = ', closest_freq

    fft_freq_delta = fft_freq[1] - fft_freq[0]
    print 'link_points.py: fft_freq_delta = ', fft_freq_delta
    num = np.floor(0.15 / fft_freq_delta)

    if num % 2 == 0:
        num += 1

    window = np.hanning(num)

    half = np.floor(num / 2.)

    closest_adj = closest - half
    big_window = np.zeros_like(x[0])
    big_window[closest_adj:closest_adj + num] = window

    closest_neg = np.where(fft_freq == -closest_freq)[0][0]
    closest_adj = closest_neg - half

    closest_adj = closest - half
    big_window = np.zeros_like(x[0])
    big_window[closest_adj:closest_adj + num] = window

    path = os.path.join(path, 'com_data_filtered_0.1.hdf5')
    new_ds = h5py.File(path, "w")

    for i in xrange(len(x)):
        this_x = x[i]
        this_y = y[i]

        ff = np.fft.fft(this_x + 1j * this_y)

        print 'link_points.py: i = ', i
        # fig = plt.figure()
        # plt.plot(fft_freq, big_window * np.abs(ff)**2)
        # plt.show()

        new = np.fft.ifft(big_window * ff)

        new_x = np.real(new)
        new_y = np.imag(new)

        # fig = plt.figure()
        # plt.plot(time, np.real(new))
        # plt.plot(time, np.imag(new))
        # plt.show()

        key_name = '%03d' % i

        #fig = plt.figure()
        #plt.plot(new_x)
        #plt.plot(this_x, 'ro', alpha = 0.3)
        #plt.show()

        single_data = np.array([x_mean[i] + new_x, y_mean[i] + new_y]).T

        dset = new_ds.create_dataset((key_name), np.shape(single_data), dtype='float', data=single_data)
    dset = new_ds.create_dataset('times', np.shape(time), dtype='float', data=time)
    new_ds.close()