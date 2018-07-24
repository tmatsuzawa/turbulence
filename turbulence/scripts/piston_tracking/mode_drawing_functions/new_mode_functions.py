import numpy as np
import h5py
import lepm.plotting.science_plot_style as sps
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import os
import lepm.plotting.colormaps as lecmaps
import pdb
import mode_drawing_functions.normal_mode_functions as nmf
import lepm.dataio as dio
import cPickle
import glob
import sys
import lepm.stringformat as sf



def load_linked_data_and_window(path_to_data):
    """

    Parameters
    ----------
    path_to_data:

    Returns
    -------
    data : list of np arrays
        [x, y, x_mean, y_mean, time]
        x_mean is an
    """
    data = h5py.File(path_to_data, 'r')
    keys = data.keys()
    x = []
    y = []
    x_mean = []
    y_mean = []
    for key in keys:
        print 'new_mf: key = ', key
        if key == 'times':
            time = np.array(data[key])
        else:
            single = np.array(data[key]).copy()
            wind = np.hanning(len(single))

            x_m = np.mean(single[:, 0])
            y_m = np.mean(single[:, 1])

            single[:, 0] = (single[:, 0] - x_m) * wind
            single[:, 1] = (single[:, 1] - y_m) * wind

            x_mean.append(x_m)
            y_mean.append(y_m)
            x.append(single[:, 0])
            y.append(single[:, 1])

    return [np.array(x), np.array(y), np.array(x_mean), np.array(y_mean), np.array(time)]


def save_mode_data(seriesdir, thres=0.005, overwrite=False, freqmin=None, freqmax=None):
    """Save the mode data if it doesn't already exist.

    Parameters
    ----------
    seriesdir : str
        The full path to the directory with all tracked cines for which to make mode decomposition movies
    thres : float
        fraction of the maximum amplitude in fft for including the mode in the saved data

    Returns
    -------

    """
    pathlist = dio.find_subdirs('201*', seriesdir)
    freqstr = ''
    if freqmin is not None:
        freqstr += '_minfreq' + sf.float2pstr(freqmin)
    if freqmax is not None:
        freqstr += '_maxfreq' + sf.float2pstr(freqmax)

    for path in pathlist:
        print 'building modes for ', path
        fn = os.path.join(path, 'com_data.hdf5')
        outfn = path + 'modes' + freqstr + '.pkl'
        outfn_traces = path + 'modes_traces' + freqstr + '.pkl'
        if not glob.glob(outfn) or not glob.glob(outfn_traces) or overwrite:
            # Note that fft_x,y are NP x #modes complex arrays
            print 'mode_drawing_functions.new_mode_functions.make_mode_movie(): loading data from ', fn
            data = load_linked_data_and_window(fn)
            tp, fft_x, fft_y, freq = nmf.ffts_and_add(data)

            high_power_inds = nmf.find_peaks(tp, percent_max=thres)

            if freqmin is not None:
                tmp_inds = np.argwhere(freq[high_power_inds] > freqmin)
                high_power_inds = high_power_inds[tmp_inds]
            if freqmax is not None:
                tmp_inds = np.argwhere(freq[high_power_inds] < freqmax)
                high_power_inds = high_power_inds[tmp_inds]

            # get the rest positions of the gyros, xy
            # xydata = load_linked_data_and_window(fn)
            meanx_arr, meany_arr = data[2], data[3]

            # Create dictionary of all the modes
            mode_data = {'xy': np.dstack((meanx_arr, meany_arr))[0]}
            mode_traces = {'xy': np.dstack((meanx_arr, meany_arr))[0]}

            for i in xrange(len(high_power_inds)):
                if i % 10 == 0:
                    print 'Constructing data for mode #', i

                x_traces, y_traces, max_mag = get_mode_drawing_data(fft_x, fft_y, freq, high_power_inds[i])
                # Get total magnitude in this maode (different from max_mag which is biggest magnitude of a single site)
                # print 'np.shape(fft_x) = ', np.shape(fft_x)
                # print 'high_power_inds = ', high_power_inds
                # Note that fft_x,y are  NP x #modes complex arrays
                tot_mag = np.sum(np.sqrt(np.abs(fft_x[:, high_power_inds[i]]) ** 2 +
                                         np.abs(fft_y[:, high_power_inds[i]]) ** 2))

                x_traces = np.array(x_traces)
                y_traces = np.array(y_traces)
                mode_data[i] = {'max_mag': max_mag,
                                'tot_mag': tot_mag,
                                'fft_x': fft_x[:, high_power_inds[i]],
                                'fft_y': fft_y[:, high_power_inds[i]],
                                'freq': freq[high_power_inds[i]],
                                }
                mode_traces[i] = {'x_traces': x_traces,
                                  'y_traces': y_traces,
                                  'freq': freq[high_power_inds[i]],
                                  'max_mag': max_mag,
                                  'tot_mag': tot_mag,
                                  }

                # print 'np.shape(fft_x) = ', np.shape(fft_x[:, i])
            # save the dictionary in pickle
            print 'saving the dictionary here: ' + outfn
            with open(outfn, "wb") as fn:
                cPickle.dump(mode_data, fn)

            with open(outfn_traces, "wb") as fn:
                cPickle.dump(mode_traces, fn)
        else:
            print 'Found mode data on disk, skipping...'


def get_mode_drawing_data(fft_x, fft_y, freq_array, max_index, **kwargs):
    """

    Parameters
    ----------
    fft_x : NP x #modes complex array
        The complex displacement for each x component of motion for every gyro (row) and frequency (column)
    fft_y : NP x #modes complex array
        The complex displacement for each y component of motion for every gyro (row) and frequency (column)
    freq_array : #modes float array
        the frequencies of normal modes, indexed by max_index
    max_index : int
        The index of the peak in the normal mode spectrum to consider
    **kwargs : 'norm' or could make other possible keyword arguments

    Returns
    -------
    """
    freq = freq_array[max_index]
    NN = len(fft_x)
    t = np.arange(41) / (abs(freq) * 40)

    mags = []
    x_traces = []
    y_traces = []
    num_gyros = len(fft_x[:, 0])
    for j in range(num_gyros):
        real_part_x = np.real(fft_x[j][max_index])
        imaginary_part_x = np.imag(fft_x[j][max_index])

        real_part_y = np.real(fft_y[j][max_index])
        imaginary_part_y = np.imag(fft_y[j][max_index])

        alpha_x = np.arctan2(imaginary_part_x, real_part_x)
        alpha_y = np.arctan2(imaginary_part_y, real_part_y)

        x_trace = np.real(abs(fft_x[j][max_index]) * np.exp(1j * (2 * np.pi * freq * t + alpha_x)))
        y_trace = np.real(abs(fft_y[j][max_index]) * np.exp(1j * (2 * np.pi * freq * t + alpha_y)))

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

    return np.array(x_traces), np.array(y_traces), max_mag


def outer_hexagon(size):
    """Return hexagonal polygon of size 'size' and the string 'hexagon'
    """
    return np.array([(size * np.cos(i * np.pi / 3 + np.pi / 2),
                      size * np.sin(i * np.pi / 3 + np.pi / 2)) for i in range(6)]), 'hexagon'


def draw_mode(x_traces, y_traces, coords, ff, axes, ii, mark_f, amp=10, output_dir='./', draw_polygon=False,
              cmap='isolum_rainbow', semilog=True):
    """Draw modes from experimental data

    Parameters
    ----------
    x_traces : #gyros x #timesteps in normal mode (typically 80) float array
        x displacements making a finely spaced polygon that traces out an ellipse for each gyro
    y_traces : #gyros x #timesteps in normal mode (typically 80) float array
        y displacements making a finely spaced polygon that traces out an ellipse for each gyro
    coords : #gyros x 2 float array
        The coordinates of the gyroscopes
    ff : [frequencies, total_power]
        The data to plot in the spectrum header
    axes :
    ii :
    mark_f : float
        The frequency whose mode we are drawing
    output_dir :
    draw_polygon :
    cmap :

    Returns
    -------
    """
    num_gyros = len(x_traces)
    ax_mode = axes[0]
    ax_freq = axes[1]

    coords[:, 0] = coords[:, 0] - np.mean(coords[:, 0])
    coords[:, 1] = coords[:, 1] - np.mean(coords[:, 1])

    plt.sca(ax_freq)
    plt.semilogy(ff[0], ff[1], 'k', linewidth=0.8)
    plt.plot([mark_f, mark_f], [plt.ylim()[0], plt.ylim()[1]], 'r', linewidth=1)
    plt.xlim(0.0, 6.)

    plt.sca(ax_mode)
    ax_mode.axis('off')

    x0 = []
    y0 = []
    colors = []
    patch = []
    for j in range(num_gyros):
        x_t = coords[j, 0] + amp * x_traces[j]
        y_t = coords[j, 1] + amp * y_traces[j]

        poly_points = np.array([x_t, y_t]).T
        polygon = Polygon(poly_points, True)

        x0.append(x_t[0])
        y0.append(y_t[0])

        colors.append((np.arctan2(y_t[0] - coords[j, 1], x_t[0] - coords[j, 0])) % (2 * np.pi))
        # colors.append((np.arctan2(y_t[0], x_t[0])) % (2 * np.pi))

        patch.append(polygon)

    xmin = 1.3 * min(coords[:, 0])
    xmin = 1.3 * min(np.min(coords[:, 1]), xmin)
    xmax = 1.3 * max(coords[:, 0])
    xmax = 1.3 * max(np.max(coords[:, 1]), xmax)

    if cmap not in plt.colormaps():
        lecmaps.register_colormap(cmap)

    p = PatchCollection(patch, cmap='isolum_rainbow', alpha=0.7, zorder=1, linewidth=0.7)
    p.set_array(np.array(colors))
    p.set_clim([0., np.pi * 2.])
    # print 'new_mode_functions: colors = ', colors
    plt.xlim(xmin, xmax)
    plt.ylim(xmax, xmin)
    ax = plt.gca()
    scat_fg = ax.scatter(x0, y0, c='k', s=1)  # this is the part that puts a dot a t=0 point

    ax.add_collection(p)
    plt.gca().set_aspect(1)
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.gca().xaxis.set_major_locator(plt.NullLocator())

    sl = abs((coords[:, 0].min() - coords[:, 0].max()) / 2)

    points, h = outer_hexagon(1.25 * sl)

    bg = [Polygon(points, True), Polygon(points, True)]
    bg = PatchCollection(bg, facecolors='#E8E8E8', zorder=0, linewidth=1)
    if draw_polygon:
        ax.add_collection(bg)
    plt.title('$\Omega$ = %0.2f Hz' % mark_f)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plt.savefig(output_dir + '/%03d.png' % ii)

    plt.close('all')


