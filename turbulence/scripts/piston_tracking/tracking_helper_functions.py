import os
import sys
from settings import tracking_settings
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt

'''Auxiliary functions for the tracking code in Hough_track.py'''


def dump_pickled_data(output_dir, filename, data):
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    of = open(output_dir + '/' + filename + '.pickle', 'wb')
    pickle.dump(data, of, pickle.HIGHEST_PROTOCOL)
    of.close()


def find_files_by_extension(root_dir, ext, tot=False):
    filenames = []
    for root, dirs, files in sorted(os.walk(root_dir)):
        for file in files:
            if file.endswith(ext):
                if tot == False:
                    filenames.append(file)
                else:
                    filenames.append(root + '/' + file)
    return filenames


def set_output_directory():
    if 'output_dir' in tracking_settings:
        output_dir = tracking_settings['output_dir']
    else:
        output_dir = './'

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    return output_dir


def shift(array):
    """Subtract mean from an array"""
    return array - np.mean(array)


def fft_on_data(dat, output_dir):
    """Performs a fourier transform on motor tracked data. colors gyroscopes by speed in an image if maxima of
    frequencies are over 100

    Parameters
    ----------
    dat : position vs time for each gyroscope in each frame
    output_dir : string specifying output directory

    Returns
    -------

    """
    t_dat = dat.T[0]

    print t_dat

    min_t = min(t_dat)

    num_gyros = len(t_dat[np.where(t_dat == min_t)])

    x_dat = dat.T[1]
    y_dat = dat.T[2]

    partitions = 1

    num_time_steps = len(x_dat) / num_gyros
    ind = np.array([k * num_gyros for k in xrange(num_time_steps)])
    t_f = np.fft.fftfreq(len(ind), 1)

    tot_power = np.zeros_like(t_f)
    m_f = []
    fft_list = []
    coords = []

    # get the data ready
    output_dir_1 = output_dir + 'fourier/'
    output_dir_old = output_dir
    if not os.path.exists(output_dir_1):
        os.mkdir(output_dir_1)
    copy_dir = output_dir

    for u in range(1):  # data can be partitioned, but currently I am just putting everything in one partition.

        num_in_partition = np.floor(num_time_steps / partitions)

        for j in range(num_gyros):
            pp = u * num_time_steps
            output_dir = os.path.join(output_dir_1, 'gy_%d/' % j)
            if not os.path.exists(output_dir): os.mkdir(output_dir)

            ind = np.array([k * num_gyros + j for k in range(num_time_steps)])

            x_gy_full = shift(x_dat[ind])
            y_gy_full = shift(y_dat[ind])
            t_gy_full = t_dat[ind]

            x_gy_full = x_gy_full[:num_in_partition * partitions]
            y_gy_full = y_gy_full[:num_in_partition * partitions]
            t_gy_full = t_gy_full[:num_in_partition * partitions]
            for u in range(partitions):

                x_gy = x_gy_full[u * num_in_partition:(u + 1) * num_in_partition]
                y_gy = y_gy_full[u * num_in_partition:(u + 1) * num_in_partition]
                t_gy = t_gy_full[u * num_in_partition:(u + 1) * num_in_partition]

                wind = np.hanning(len(t_gy))

                coords.append([np.mean(x_dat[ind]), np.mean(y_dat[ind])])

                fft_cylindrical_a = np.fft.fft(wind * np.array(x_gy + 1j * y_gy))
                fft_cylindrical_negative = np.fft.fft(wind * np.array(x_gy - 1j * y_gy))
                fft_cylindrical = abs(fft_cylindrical_a) ** 2
                fft_cylindrical_n = abs(fft_cylindrical_negative) ** 2
                fft_x = np.fft.fft(wind * np.array(x_gy))
                fft_y = np.fft.fft(wind * np.array(y_gy))
                fft_freq = np.fft.fftfreq(len(t_gy), t_gy[1] - t_gy[0])

                dump_pickled_data(output_dir, 'x_gy_%01d' % u, x_dat[ind])
                dump_pickled_data(output_dir, 'y_gy_%01d' % u, y_dat[ind])
                dump_pickled_data(output_dir, 'fft_x_%01d' % u, np.array([fft_freq, fft_x]).T)
                dump_pickled_data(output_dir, 'fft_y_%01d' % u, np.array([fft_freq, fft_y]).T)
                dump_pickled_data(output_dir, 'fft_complex_%01d_positive' % u,
                                  np.array([fft_freq, fft_cylindrical_a]).T)
                dump_pickled_data(output_dir, 'fft_complex_%01d_negative' % u,
                                  np.array([fft_freq, fft_cylindrical_negative]).T)
                dump_pickled_data(output_dir, 'coords_%01d.pickle' % u, coords[j])
                fft_list.append(fft_cylindrical_a)

                max_xf = max(fft_cylindrical)

                maximum_f = fft_freq[np.where(fft_cylindrical == max_xf)[0][0]]

                if True:  # j ==0:
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    plt.plot(np.array(t_gy), x_gy)
                    plt.plot(np.array(t_gy), y_gy)

                    plt.savefig(output_dir + 'data_xy_%01d.png' % u)
                    print output_dir + 'data_xy_%01d.png' % u

                    lab = ['max of x + iy fft = %0.3f ' % maximum_f, '', '']
                    fig3 = plt.figure()
                    ax = fig3.add_subplot(1, 1, 1)
                    plt.plot(fft_freq[1:], abs(fft_cylindrical[1:]), label=lab[0])
                    plt.plot(fft_freq[1:], abs(fft_cylindrical_n[1:]), label=lab[0])
                    plt.xlabel('Freq (Hz)', fontsize=11)
                    plt.ylim(0, max_xf * 1.2)
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, fancybox=True, fontsize=11, loc=1)
                    # plt.yscale('log')
                    plt.xlim(0, 3.5)
                    plt.savefig(output_dir + 'fft_cyl_%01d.png' % u)

                    plt.close()

                if max(fft_freq > 100):
                    if abs(maximum_f) < 10:
                        selected_f = fft_freq[fft_freq < -100]
                        selected = fft_cylindrical[fft_freq < -100]

                        max_xf = max(selected)
                        maximum_f = selected_f[np.where(selected == max_xf)[0][0]]

                    if abs(maximum_f) > 150 and abs(maximum_f) < 3200:
                        m_f = list(m_f)
                        m_f.append([j, maximum_f])

                plt.close()

                if j == 0:
                    tot_power = np.zeros_like(fft_cylindrical)
                else:

                    tot_power += fft_cylindrical

        if max(fft_freq) > 100:

            if len(m_f) > 1:
                dump_pickled_data(copy_dir, '/speed', np.array(m_f))
                m_f = np.array(m_f).T[1]
                fig55 = plt.figure()
                ax = fig55.add_subplot(111)
                n, bins, patches = ax.hist(abs(np.array(m_f)), 100, normed=False, facecolor='green', alpha=0.75)
                plt.savefig(copy_dir + '/speed.png')
                m_f = np.array(m_f)
                plt.close()

                print output_dir_old + 'com_data.pickle'
                color_by_speed(output_dir_old + 'com_data.pickle', output_dir_old)
