import numpy as np
import h5py
import lepm.plotting.science_plot_style as sps
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import os


def load_linked_data_and_window(path_to_data):
    """

    Parameters
    ----------
    path_to_data:

    Returns
    -------
    data : list of np arrays
        [x, y, x_mean, y_mean, time]
    """
    data = h5py.File(path_to_data, 'r')
    keys = data.keys()
    x = []
    y = []
    x_mean = []
    y_mean = []
    for key in keys:
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


def get_mode_drawing_data(fft_x, fft_y, freq_array, max_index, **kwargs):
    freq = freq_array[max_index]
    NN = len(fft_x)
    t = np.arange(41) * 1 / (abs(freq) * 40)

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


def outer_hexagon(size): return np.array(
    [(size * np.cos(i * np.pi / 3 + np.pi / 2), size * np.sin(i * np.pi / 3 + np.pi / 2)) for i in range(6)]), 'hexagon'


def draw_mode(x_traces, y_traces, coords, ff, axes, ii, mark_f, fac=5, output_dir='./'):
    num_gyros = len(x_traces)
    ax_mode = axes[0]
    ax_freq = axes[1]

    coords[:, 0] = coords[:, 0] - np.mean(coords[:, 0])
    coords[:, 1] = coords[:, 1] - np.mean(coords[:, 1])

    plt.sca(ax_freq)
    plt.plot(ff[0], ff[1], 'k', linewidth=0.8)
    plt.plot([mark_f, mark_f], [0, plt.ylim()[1]], 'r', linewidth=1)
    plt.xlim(1.0, 3.)

    plt.sca(ax_mode)
    ax_mode.axis('off')

    x0 = []
    y0 = []
    colors = []
    patch = []
    for j in range(num_gyros):
        x_t = coords[j, 0] + fac * x_traces[j]
        y_t = coords[j, 1] + fac * y_traces[j]

        poly_points = np.array([x_t, y_t]).T
        polygon = Polygon(poly_points, True)

        x0.append(x_t[0])
        y0.append(y_t[0])

        colors.append((np.arctan2(y_t[0] - coords[j, 1], x_t[0] - coords[j, 0])) % (2 * np.pi))

        patch.append(polygon)

    xmin = 1.3 * min(coords[:, 0])
    ymin = 1.3 * min(coords[:, 1])
    xmax = 1.3 * max(coords[:, 0])
    ymax = 1.3 * max(coords[:, 1])
    p = PatchCollection(patch, cmap='isolum_rainbow', alpha=0.7, zorder=1, linewidth=0.7)
    p.set_array(np.array(colors))
    plt.xlim(xmin, xmax)
    plt.ylim(ymax, ymin)
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
    ax.add_collection(bg)
    plt.title('$\Omega$ = %0.2f Hz' % mark_f)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plt.savefig(output_dir + '/%03d.png' % ii)

    plt.close()


def ffts_and_add(data):


    # the input data should be by gyroscope
    x_data_all = data[0]
    y_data_all = data[1]
    times = data[4]
    dt = times[1] - times[0]

    fft_list_x = []
    fft_list_y = []

    freq = np.fft.fftfreq(len(times), d=dt)
    si = np.argsort(freq)
    freq = freq[si]
    inds = np.where(freq > 0.4)
    freq = freq[inds]
    tot_power = np.zeros_like(freq)
    for i in xrange(len(x_data_all)):
        x_data = x_data_all[i]
        y_data = y_data_all[i]

        fft_x = np.fft.fft(x_data)
        fft_y = np.fft.fft(y_data)
        fft_x = fft_x[si]
        fft_y = fft_y[si]
        fft_x = fft_x[inds]
        fft_y = fft_y[inds]
        tot_power += abs(fft_x) ** 2 + abs(fft_y) ** 2
        fft_list_x.append(fft_x)
        fft_list_y.append(fft_y)

    return tot_power, np.array(fft_list_x), np.array(fft_list_y), freq
