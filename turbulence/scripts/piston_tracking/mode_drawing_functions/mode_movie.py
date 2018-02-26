import numpy as np
# import h5py
import os
import sys
# import matplotlib.pyplot as plt
import normal_mode_functions as nmf
import new_mode_functions as new_mf
import lepm.plotting.science_plot_style as sps
import lepm.data_handling as dh
import lepm.dataio as dio
import glob
import lepm.plotting.movies as movies
import lepm.stringformat as sf
import cPickle

"""Script to run mode decomposition of experimental data"""


def make_mode_movie(seriesdir, amp=50, semilog=True, freqmin=None, freqmax=None, overwrite=False, percent_max=0.01):
    """

    Parameters
    ----------
    seriesdir : str
        The full path to the directory with all tracked cines for which to make mode decomposition movies
    amp : float
        The amplification factor for the displacements in the movie output
    semilog : bool
        plot the FFT intensity vs frequency in log-normal scale
    freqmin : float
        The minimum frequency to plot
    freqmax : float
        The maximum frequency to plot
    overwrite : bool
        Overwrite the saved images/movies if they exist

    Returns
    -------
    """
    pathlist = dio.find_subdirs('20*', seriesdir)
    freqstr = ''
    if freqmin is not None:
        freqstr += '_minfreq' + sf.float2pstr(freqmin)
    if freqmax is not None:
        freqstr += '_maxfreq' + sf.float2pstr(freqmax)

    for path in pathlist:
        movname = path + 'modes_amp{0:0.1f}'.format(amp).replace('.', 'p') + freqstr + '.mov'
        movexist = glob.glob(movname)
        print 'mode_movie: movexist = ', movexist

        if not movexist or overwrite:
            # If the movie does not exist yet, make it here
            print 'building modes for ', path
            fn = os.path.join(path, 'com_data.hdf5')
            print 'mode_drawing_functions.mode_movie.make_mode_movie(): loading data from ', fn
            data = new_mf.load_linked_data_and_window(fn)

            tp, fft_x, fft_y, freq = nmf.ffts_and_add(data)
            high_power_inds = nmf.find_peaks(tp, percent_max=percent_max)

            # Check how many mode images have been done
            modespngs = glob.glob(path + 'modes' + freqstr + '/*.png')
            modespngs_traces = glob.glob(path + 'modes_traces' + freqstr + '/*.png')

            # If we haven't made images of all the modes, do that here
            print 'mode_movie.make_mode_movie(): len(modespngs) = ', len(modespngs)
            print 'mode_movie.make_mode_movie(): len(high_power_inds) = ', len(high_power_inds)

            if len(modespngs) < len(high_power_inds) or overwrite:
                # Check if mode pickle is saved
                modesfn_traces = path + 'modes_trackes' + freqstr + '.pkl'
                globfn = glob.glob(modesfn_traces)
                if globfn:
                    with open(globfn[0], "rb") as fn:
                        mode_data = cPickle.load(fn)

                    coords = mode_data['xy']
                    mode_data = dh.removekey(mode_data, 'xy')

                    for i in mode_data:
                        if i % 10 == 0:
                            print 'mode_movie.make_mode_movie(): Creating mode image #', i

                        x_traces = mode_data[i]['x_traces']  # sf.float2pcstr(freq[high_power_inds[i]], ndigits=8)]
                        y_traces = mode_data[i]['y_traces']

                        fig = sps.figure_in_mm(120, 155)
                        ax_mode = sps.axes_in_mm(10, 10, 100, 100)
                        ax_freq = sps.axes_in_mm(10, 120, 100, 30)

                        axes = [ax_mode, ax_freq]

                        new_mf.draw_mode(x_traces, y_traces, coords, [freq, tp], axes, i, freq[high_power_inds[i]],
                                         output_dir=os.path.join(path, 'modes'), amp=amp, semilog=semilog)
                else:
                    # The mode data is not saved, so we must create it as we go along
                    # data = new_mf.load_linked_data_and_window(fn)
                    coords = np.array([data[2], data[3]]).T

                    for i in xrange(len(high_power_inds)):
                        if i % 10 == 0:
                            print 'mode_movie.make_mode_movie(): Creating mode image #', i

                        x_traces, y_traces, max_mag = new_mf.get_mode_drawing_data(fft_x, fft_y, freq, high_power_inds[i])

                        x_traces = np.array(x_traces)
                        y_traces = np.array(y_traces)

                        fig = sps.figure_in_mm(120, 155)
                        ax_mode = sps.axes_in_mm(10, 10, 100, 100)
                        ax_freq = sps.axes_in_mm(10, 120, 100, 30)

                        axes = [ax_mode, ax_freq]

                        new_mf.draw_mode(x_traces, y_traces, coords, [freq, tp], axes, i, freq[high_power_inds[i]],
                                         output_dir=os.path.join(path, 'modes'), amp=amp, semilog=semilog)

            # Obtain imagename and moviename, create movie if it doesn't exist
            modesfn = glob.glob(path + 'modes' + freqstr + '/*.png')
            imagename_split = modesfn[0].split('/')[-1].split('.png')[0]
            try:
                test = int(imagename_split)
                indexsz = len(imagename_split)
                imgname = path + 'modes' + freqstr + '/'
            except:
                print 'mode_movie.make_mode_movie(): imagename_split = ', imagename_split
                print 'mode_movie.make_mode_movie(): indexsz = ', len(imagename_split)
                raise RuntimeError('Imagename is not just an int -- write code to allow ')

            movies.make_movie(imgname, movname, indexsz=str(indexsz), framerate=10)


if __name__ == '__main__':
    seriesdir = '/Volumes/labshared2/noah/test3gyro_a3p220in_1p6in_tracking/'
    make_mode_movie(seriesdir)