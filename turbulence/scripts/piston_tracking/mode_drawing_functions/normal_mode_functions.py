import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
# import cPickle as pickle
# import Motor_tracking_code.motor_track_functions as mtf
# import Motor_tracking_code.motor_track as mt
# import os.path
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import lepm.plotting.science_plot_style as sps
# from scipy.signal import argrelextrema
# import isolum_rainbow


def mode_drawing_data(fft_x, fft_y, num_gyros, freq, NN, mi, **kwargs):
    """

    fft_x:
    fft_y:
    num_gyros:
    freq:
    NN:
    mi:
    kwargs:

    """
    t = np.arange(41) * 1 / (abs(freq) * 40)

    mags = []
    x_traces = []
    y_traces = []
    for j in range(num_gyros):
        real_part_x = np.real(fft_x[j][mi])
        imaginary_part_x = np.imag(fft_x[j][mi])

        real_part_y = np.real(fft_y[j][mi])
        imaginary_part_y = np.imag(fft_y[j][mi])

        alpha_x = np.arctan2(imaginary_part_x, real_part_x)
        alpha_y = np.arctan2(imaginary_part_y, real_part_y)

        x_trace = np.real(abs(fft_x[j][mi]) * np.exp(1j * (2 * np.pi * freq * t + alpha_x)))
        y_trace = np.real(abs(fft_y[j][mi]) * np.exp(1j * (2 * np.pi * freq * t + alpha_y)))

        x_traces.append(x_trace * (2. / NN))
        y_traces.append(y_trace * (2. / NN))

        mags.append(np.sqrt(abs(x_trace * (2. / NN)) ** 2 + abs(y_trace * (2. / NN)) ** 2))

    max_mag = 2 * max(np.array(mags).flatten())

    if 'norm' in kwargs:
        norm = kwargs['norm']
        if norm:
            x_traces = np.array(x_traces) / max_mag
            y_traces = np.array(y_traces) / max_mag
        else:

            x_traces = np.array(x_traces)
            y_traces = np.array(y_traces)

    return x_traces, y_traces, max_mag
#
#
# def draw_mode(x_traces, y_traces, num_gyros, coords, ii=0, output_dir='./', **kwargs):
#     """Draw the normal mode of the gyroscopic network based on the fft data
#
#     Parameters
#     ----------
#     x_traces:
#     y_traces:
#     num_gyros:
#     coords:
#     ii:
#     output_dir:
#     kwargs:
#     :return:
#     """
#     if 'freq_array' in kwargs:
#         freq_array = kwargs['freq_array']
#
#         if 'freq' in kwargs:
#             freq = kwargs['freq']
#             plt.title('%0.2f Hz' % freq)
#         if 'factor' in kwargs:
#             fac = kwargs['factor']
#         else:
#             fac = 80
#
#         fig = sps.figure_in_mm(120, 155)
#         ax_mode = sps.axes_in_mm(10, 10, 100, 100)
#         ax_freq = sps.axes_in_mm(10, 120, 100, 30)
#
#         plt.sca(ax_freq)
#         if len(freq_array) == 1:
#             freq_array = list(freq_array)
#             freq_array.append(0)
#
#         if not 'power_array' in kwargs:
#             n, bins, patches = plt.hist(freq_array, 100, facecolor='#80D080')
#             plt.ylim(0, 3)
#         else:
#             pa = kwargs['power_array']
#
#             plt.plot(freq_array, pa)
#
#         (f_mark,) = plt.plot([freq, freq], plt.ylim(), '-r')
#         # plt.xlim(0,6)
#
#         patch = []
#         x0 = []
#         y0 = []
#         colors = np.zeros(num_gyros)
#
#         plt.sca(ax_mode)
#
#         if 'freq' in kwargs:
#             freq = kwargs['freq']
#             ax_mode.set_title('Mode %0d %0.2f Hz' % (ii, freq))
#         for j in range(num_gyros):
#             x_t = coords[j, 0] + fac * x_traces[j]
#             y_t = coords[j, 1] + fac * y_traces[j]
#
#             poly_points = np.array([x_t, y_t]).T
#             polygon = Polygon(poly_points, True)
#
#             x0.append(x_t[0])
#             y0.append(y_t[0])
#
#             mag = np.sqrt(x_t[0] ** 2 + y_t[0] ** 2)
#             anglez = (np.arctan2(y_t[0] - coords[j, 1], x_t[0] - coords[j, 0])) % (2 * np.pi)
#
#             colors[j] = anglez
#
#             patch.append(polygon)
#
#         xmin = 1.2 * min(coords[:, 0])
#         ymin = 1.2 * min(coords[:, 1])
#         xmax = 1.2 * max(coords[:, 0])
#         ymax = 1.2 * max(coords[:, 1])
#         p = PatchCollection(patch, cmap='isolum_rainbow', alpha=0.5)
#         p.set_array(np.array(colors))
#         plt.xlim(xmin, xmax)
#         plt.ylim(ymax, ymin)
#         ax = plt.gca()
#         scat_fg = ax.scatter(x0, y0, c='k', s=1)  # this is the part that puts a dot a t=0 point
#         ax.add_collection(p)
#         plt.gca().set_aspect(1)
#         plt.gca().yaxis.set_major_locator(plt.NullLocator())
#         plt.gca().xaxis.set_major_locator(plt.NullLocator())
#
#     else:
#         fig = sps.figure_in_mm(120, 120)
#         patch = []
#         x0 = []
#         y0 = []
#         colors = np.zeros(num_gyros)
#
#         for j in range(num_gyros):
#             x_t = coords[j, 0] + fac * x_traces[j]
#             y_t = coords[j, 1] + fac * y_traces[j]
#
#             poly_points = np.array([x_t, y_t]).T
#             polygon = Polygon(poly_points, True)
#
#             x0.append(x_t[0])
#             y0.append(y_t[0])
#
#             mag = np.sqrt(x_t[0] ** 2 + y_t[0] ** 2)
#             anglez = (np.arctan2(y_t[0] - coords[j, 1], x_t[0] - coords[j, 0])) % (2 * np.pi)
#
#             colors[j] = anglez
#
#             patch.append(polygon)
#
#         xmin = 1.2 * min(coords[:, 0])
#         ymin = 1.2 * min(coords[:, 1])
#         xmax = 1.2 * max(coords[:, 0])
#         ymax = 1.2 * max(coords[:, 1])
#         p = PatchCollection(patch, cmap='isolum_rainbow', alpha=0.8)
#         p.set_array(np.array(colors))
#         plt.xlim(xmin, xmax)
#         plt.ylim(ymax, ymin)
#         ax = plt.gca()
#         scat_fg = ax.scatter(x0, y0, c='k', s=1)  # this is the part that puts a dot a t=0 point
#         ax.add_collection(p)
#         plt.gca().set_aspect(1)
#         plt.gca().yaxis.set_major_locator(plt.NullLocator())
#         plt.gca().xaxis.set_major_locator(plt.NullLocator())
#
#         if 'freq' in kwargs:
#             freq = kwargs['freq']
#             plt.title('%0.2f Hz' % freq)
#
#     plt.savefig(output_dir + '/%03d.png' % ii)
#
#     plt.close()


def find_peaks(data_array, percent_max=0.01, smooth=False):
    """

    Parameters
    ----------
    data_array :
    percent_max:
    smooth:

    Returns
    -------
    max_inds : the indices of the normal modes which have amplitude of at least percent_max * max_amplitude
    """
    # print 'data_array = ', data_array
    maxes = np.array(range(len(data_array)))  # np.array(argrelextrema(data_array, np.greater))
    # maxes = np.array(argrelextrema(data_array, np.greater))
    max_max_peak = max(data_array[maxes].flatten())
    max_max_index = np.where(data_array == max_max_peak)[0]

    m_inds = np.where(data_array[maxes] > percent_max * max_max_peak)  # frequency of peaks

    max_inds = maxes[m_inds]

    return max_inds


def save_files_for_kitaev(fft_x, fft_y, freq):
    """

    Parameters
    ----------
    fft_x :
    fft_y :
    freq :
    """
    NP = len(fft_x[:, 0])

    print 'number of gyros is', NP
    raise RuntimeError('Have not written this function yet')


def ffts_and_add(data, cutoff=0.4):
    """Calculate the fft

    Parameters
    ----------
    data : [x, y, x_mean, y_mean, time]
    cutoff : float
        Lowest allowed frequency to keep

    Returns
    -------
    tot_power : n x 1 float array
        The total power in each fourier mode
    fft_x : NP x #modes complex array
        The complex displacement for each x component of motion for every gyro (row) and frequency (column)
    fft_y : NP x #modes complex array
        The complex displacement for each y component of motion for every gyro (row) and frequency (column)
    freq : n x 1 float array
        The frequencies of the allowed fourier transform
    """
    # the input data should be by gyroscope
    x_data_all = data[0]
    y_data_all = data[1]
    # print 'normal_mode_functions: data= ', data
    times = data[4]
    dt = times[1] - times[0]
    print 'normal_mode_functions: dt = ', dt, '\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n'
    if (np.diff(times) != dt).all():
        raise RuntimeError('dt is not constant in this data, exiting.')

    fft_list_x = []
    fft_list_y = []

    freq = np.fft.fftfreq(len(times), d=dt)
    si = np.argsort(freq)
    freq = freq[si]
    inds = np.where(freq > cutoff)
    freq = freq[inds]
    tot_power = np.zeros_like(freq)
    for dmyi in xrange(len(x_data_all)):
        x_data = x_data_all[dmyi]
        y_data = y_data_all[dmyi]

        # print 'normal_mode_functions: plotting...'
        # plt.plot(times, np.real(x_data), 'o-')
        # plt.plot(times, np.real(y_data), 'o-')
        # plt.show()
        # plt.clf()

        fft_x = np.fft.fft(x_data)[si][inds]
        fft_y = np.fft.fft(y_data)[si][inds]
        tot_power += abs(fft_x) ** 2 + abs(fft_y) ** 2

        # print 'normal_mode_functions: plotting...'
        # plt.plot(freq, np.real(fft_x), 'o-')
        # plt.plot(freq, np.imag(fft_x), 'o-')
        # plt.plot(freq, np.real(fft_y), 'o-')
        # plt.plot(freq, np.imag(fft_y), 'o-')
        # plt.show()
        # plt.clf()

        fft_list_x.append(fft_x)
        fft_list_y.append(fft_y)

    return tot_power, np.array(fft_list_x), np.array(fft_list_y), freq
