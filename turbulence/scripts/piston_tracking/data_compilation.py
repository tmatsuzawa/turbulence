"""Read com data.hdf5 from each folder, and plot position vs time and velocity vs time, and calculates effective velocity
"""
import sys
sys.path.append('/Users/stephane/Documents/git/takumi/turbulence/turbulence/scripts/piston_tracking/')
import experiment_movies.experiment_movie as mnf
import Hough_track as ht
#import turbulence.scripts.piston_tracking.experiment_movies.experiment_movie as mnf

import numpy as np
import time
from mode_drawing_functions.get_speed_and_plot import get_speed_and_plot_withresolution
import mode_drawing_functions.mode_movie as modemov
import mode_drawing_functions.new_mode_functions as nmf

import library.display.graph as graph
import library.basics.formatstring as fs
import library.basics.formatarray as fa
import library.tools.process_data as process
import library.tools.pickle_rw as pickle_rw
import library.tools.rw_data as rw

import argparse
import os
import matplotlib.pyplot as plt
import h5py
from scipy.signal import savgol_filter
from scipy import integrate

parser = argparse.ArgumentParser(description='Track motion of motor')
parser.add_argument('-inputdir', '--inputdir', help='input dir', type=str)
# Experimental parameters
parser.add_argument('-fx', '--fx', help='Conversion factor: ', type=float, default=1.)
parser.add_argument('-freq', '--freq', help='Frequency of piston motion: ', type=float, default=1.)

# Analysis settings
parser.add_argument('-shift', '--shift', help='Number of elements shifted for Fig3', type=int, default=100)
parser.add_argument('-shiftper', '--shiftper', help='Shift the data arrays by this much percentage forward when plotting', type=float, default=0.2)
parser.add_argument('-reverse', '--reverse', help='Track videos in reverse order that they appear in dir',
                    action='store_true')
parser.add_argument('-start', '--start', help='frames used to analyze [start, total_frame_number - end]', type=int, default=0)
parser.add_argument('-end', '--end', help='frames used to analyze (start, total_frame_number - end]', type=int, default=0)
parser.add_argument('-overwrite', '--overwrite', help='Overwrite previous tracking results', action='store_true')

# Plot effective velocity vs commanded velocity using all cines in the selected directory
parser.add_argument('-datacomp', '--datacomp', help='Plot effective velocity vs commanded velocity using all cines in the selected directory',
                    type=bool, default=True)

args = parser.parse_args()


ay_threshold = 00. # Threshold to detect a lump in velocity profile [mm/s2]

def detect_sign_flip(arr):
    """
    Returns indices of an 1D array where its elements flip the sign
    Parameters
    ----------
    arr

    Returns
    -------
    indices: tuple

    """
    arr = np.array(arr)
    arrsign = np.sign(arr)
    signchange = ((np.roll(arrsign, 1) - arrsign) != 0).astype(int)
    indices = np.where(signchange == 1)
    return indices

def compute_eff_velocity(vel, time, ind1, ind2):
    """ Computes effective velocity

    Parameters
    ----------
    vel
    ind1
    ind2

    Returns
    -------

    """
    if not len(vel) == len(time):
        print 'velocity array and time array have different sizes!... Continue computing effective velocity.'

    # Clip an array for computation
    vel = vel[ind1-1:ind2]
    time = time[ind1-1:ind2]
    # Prepare velocity squared array
    vel2 = [v**2 for v in vel]
    # Integrate (composite trapezoid)
    vel_int = integrate.trapz(vel, time)
    vel2_int = integrate.trapz(vel2, time)
    # Compute effective velocity
    v_eff = vel2_int / vel_int
    return v_eff

def compute_mean_velocity(vel, time, ind1, ind2):
    """ Computes effective velocity

    Parameters
    ----------
    vel
    ind1
    ind2

    Returns
    -------

    """
    if not len(vel) == len(time):
        print 'velocity array and time array have different sizes!... Continue computing effective velocity.'

    # Clip an array for computation
    vel = vel[ind1-1:ind2]
    vel_avg = np.nanmean(vel)
    return vel_avg


def get_position_from_step_hdf5(root_dir, skip_savepath=False, verbose=False, cutoff_distance=None):
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

        image_path = os.path.join(root_dir, 'gyro_path_images/')
        if not os.path.exists(image_path):
            os.mkdir(image_path)

        # image_path = os.path.join(image_path, '%03d.png' % ii)
        # fig = plt.figure()
        # ax1 = fig.add_subplot(211)
        # ax2 = fig.add_subplot(212)
        len_single = len(single_data)
        # if not skip_savepath:
        #     ax1.plot(times[:len_single], single_data[:, 0])
        #
        #     # make sign for where the particle is in space
        #     ax1.set_title(r'$x$ displacement for particle ' + str(ii) + ' at ({0:0.0f}'.format(single_data[0, 0]) +
        #                   ',{0:0.0f}'.format(single_data[0, 1]) + ')')
        #     ax2.plot(compare[:, 1], compare[:, 2], 'k.')
        #     ax2.plot(compare[ii, 1], compare[ii, 2], 'ro')
        #     ax2.axis('scaled')
        #     ax2.axis('off')
        #     ax1.set_xlabel('time [s]')
        #     ax1.set_ylabel(r'$x$ displacement')
        #     plt.savefig(image_path)
        #     plt.close()

    dset = new.create_dataset('times', np.shape(times), dtype='float', data=times)
    new.close()
    data.close()
    return times[:len_single], single_data[:, 0], single_data[:, 1]

def update_data_dict(dict, key, subkey, data):
    """
    Generate a dictionary that stores effective velocity
    Parameters
    ----------
    dict
    key: span like span5.4
    subkey: commanded velocity, str
    data: effective velocity, float

    Returns
    -------

    """
    if not key in dict:
        dict[key] = {}  # Generate a sub-dictionary
    dict[key][subkey] = data
    return dict

if __name__ == '__main__':
    start_time = time.time()
    end_time = time.time()
    from settings import tracking_settings

    print 'tracking...'
    fns, outputs = ht.run_tracking(tracking_settings, reverse=args.reverse, overwrite=args.overwrite)
    print 'done tracking...'

    data_dict, vmin_dict = {}, {}

    for i in range(len(outputs)):
        # Initialization
        x_pos_2d, y_pos_2d, u_2d, v_2d = [], [], [], []

        # Get a path to a file that stores the centroid position vs time
        if fns==[]:
            fns = outputs
        fn_noext, ext = os.path.splitext(fns[i])
        fn_noext_head, fn_noext_tail = os.path.split(fn_noext)

        # load data from the hdf5 file
        time, x_pos, y_pos = get_position_from_step_hdf5(fn_noext)
        xm_pos, ym_pos = np.mean(x_pos), np.mean(y_pos)


        # Convert px->mm
        x_pos[:] = (x_pos[:] - np.min(x_pos))* args.fx
        y_pos[:] = (y_pos[:] - np.min(y_pos)) * args.fx
        xm_pos, ym_pos = xm_pos * args.fx, ym_pos * args.fx


        #frame_rate = fs.get_float_from_str(fn_noext, '20180220_', 'fps')
        #frame_rate = ht.extract_frame_rate(fns[i])
        try:
            frame_rate = ht.extract_frame_rate(fns[i])
        except ValueError:
            frame_rate = fs.get_float_from_str(fn_noext_tail, 'frate', 'fps')
        span = fs.get_float_from_str(fn_noext_tail, 'span', 'mm')
        vp_commanded = fs.get_float_from_str(fn_noext_tail, '_v', 'mms')
        freq = fs.get_float_from_str(fn_noext_tail, 'mms_f', 'Hz')
        #freq = args.freq


        # Compute velocity from position
        u = np.gradient(x_pos) * frame_rate
        v = np.gradient(y_pos) * frame_rate

        # Piston tracking plot: x, y, u, v vs time
        title = os.path.split(fn_noext)[1]
        fig3, ax1 = graph.plot(time, x_pos, fignum=3, subplot=221, figsize=(12, 8))
        fig3, ax2 = graph.plot(time, y_pos, fignum=3, subplot=222, figsize=(12, 8))
        fig3, ax3 = graph.plot(time, u, fignum=3, subplot=223, figsize=(12, 8))
        fig3, ax4 = graph.plot(time, v, fignum=3, subplot=224, figsize=(12, 8))
        plt.suptitle(title)
        graph.labelaxes(ax1, 'Time [s]', 'X [mm]', fontsize=8)
        graph.labelaxes(ax2, 'Time [s]', 'Y [mm]', fontsize=8)
        graph.labelaxes(ax3, 'Time [s]', 'u [mm/s]', fontsize=8)
        graph.labelaxes(ax4, 'Time [s]', 'v [mm/s]', fontsize=8)
        plotfilename = os.path.join(fn_noext, 'plots/position_time')
        graph.save(plotfilename, verbose=True)

        # Piston tracking: average over multiple cycles
        print 'file name:' + fn_noext
        if not args.end==0:
            x_pos = x_pos[args.start:-args.end]
            y_pos = y_pos[args.start:-args.end]
            u = u[args.start:-args.end]
            v = v[args.start:-args.end]
            print 'Number of frames in cine: %d' % len(time)
            time = time[args.start:-args.end]
        else:
            x_pos = x_pos[args.start:]
            y_pos = y_pos[args.start:]
            u = u[args.start:]
            v = v[args.start:]
            print 'Number of frames in cine: %d' % len(time)
            time = time[args.start:]


        # Print extracted info about a movie
        print 'Number of frames used: %d' % len(time)
        print 'Frame rate: %.1f' % frame_rate
        print 'Frequency: %.1f' % freq
        print '# of cycles in movie: %d' % int(np.ceil(len(time)/frame_rate*freq))
        numcycles = int(np.ceil(len(time)/frame_rate*freq))

        x_pos_chunks = fa.array2nchunks(x_pos, numcycles)
        y_pos_chunks = fa.array2nchunks(y_pos, numcycles)
        u_chunks = fa.array2nchunks(u, numcycles)
        v_chunks = fa.array2nchunks(v, numcycles)
        time_chunks = fa.array2nchunks(time, numcycles)

        time_short = time_chunks.next()
        time_short = time_short - np.min(time_short)

        # Roll ypos
        y_pos = np.roll(y_pos, int(len(y_pos) * args.shiftper))

        for y_pos_chunk in y_pos_chunks:
            if len(y_pos_chunk) < len(time_short):
                print 'The last chunk has fewer elements than other chunks.' \
                      '-> Make the last chunk contain as many elements as other chunks...'
                y_pos_chunk = fa.extend_1darray_fill(y_pos_chunk, len(time_short), fill_value=np.nan)
            # Shift the data array for plotting sake
            y_pos_chunk = np.roll(y_pos_chunk, int(len(y_pos_chunk) * args.shiftper))
            y_pos_2d.append(y_pos_chunk)  #<- this is a list of lists

        y_pos_2d = np.concatenate(np.transpose(y_pos_2d)).ravel().reshape(len(y_pos_chunk), numcycles) #<- Now, this is 2d array.

        # Calculate average and std for position
        y_pos_mean = np.nanmean(y_pos_2d, axis=1)
        y_pos_std = np.nanstd(y_pos_2d, axis=1)

        # Plot y vs time
        fig4, ax5, color_patch5 = graph.errorfill(time_short, y_pos_mean, y_pos_std, fignum=4, subplot=211, color='b')
#        fig4, ax5, color_patch6 = graph.errorfill(time_short, y_pos, args.fx, fmt='x', fignum=4, color='C2', subplot=211)

        # Output actual stroke length and its error
        actual_span = np.max(y_pos_mean) - np.min(y_pos_mean)
        y_pos_std_at_max, y_pos_std_at_min = y_pos_std[np.argmax(y_pos_mean)], y_pos_std[np.argmin(y_pos_mean)]
        actual_span_err = np.sqrt(y_pos_std_at_max ** 2 + y_pos_std_at_min **2)  # propagation of error
        # Add text
        text = 'Average stroke length = %.2f $\pm$ %.2f mm' % (actual_span, actual_span_err)
        graph.addtext(ax5, text=text, subplot=211, option='bl', fontsize=10)
        graph.labelaxes(ax5, 'Time [s]', 'Piston position [mm]')

        for v_chunk in v_chunks:
            if len(v_chunk) < len(time_short):
                print 'The last chunk has fewer elements than other chunks.' \
                      '-> Make the last chunk contain as many elements as other chunks...'
                v_chunk = fa.extend_1darray_fill(v_chunk, len(time_short), fill_value=np.nan)
            # Shift the data array for plotting sake
            v_chunk = np.roll(v_chunk, int(len(v_chunk) * args.shiftper))
            v_2d.append(v_chunk)

        # Calculate average and std of velocity
        v_2d = np.concatenate(np.transpose(v_2d)).ravel().reshape(len(y_pos_chunk), numcycles)  # <- Now, this is 2d array.
        v_mean = np.nanmean(v_2d, axis=1)
        v_std = np.nanstd(v_2d, axis=1)
        fig4, ax7, color_patch6 = graph.errorfill(time_short, v_mean, v_std, fignum=4, subplot=212, color='b')
        plt.suptitle(title)



        # Calculate effective velocity
        ## Interpolate the data set (time_short and v_mean)
        time_short_int, v_mean_int = process.interpolate_1Darrays(time_short, v_mean)

        ## Get two points where y=0
        ### Method: Detect sign flip
        # Find a minimum value and its index
        v_min_ind, v_min = np.argmin(v_mean_int), np.amin(v_mean_int)
        # Split an array into two parts using the minimum value
        v_mean_1, v_mean_2 = v_mean_int[:v_min_ind], v_mean_int[v_min_ind:]
        # Detect a sign flip of the left array
        signflip_indices = fa.detect_sign_flip(v_mean_1)
        v_mean_left_ind = signflip_indices[-1]
        # Detect a sign flip of the right array
        signflip_indices = fa.detect_sign_flip(v_mean_2)
        v_mean_right_ind = len(v_mean_1) + signflip_indices[0]

        # Plot two points between which effective velocity is calculated
        graph.scatter([time_short_int[v_mean_left_ind]], [v_mean_int[v_mean_left_ind]], fignum=4, subplot=212, marker='x', color='r')
        graph.scatter([time_short_int[v_mean_right_ind]], [v_mean_int[v_mean_right_ind]], fignum=4, subplot=212, marker='x', color='r')

        # Compute effective velocity
        veff = compute_eff_velocity(v_mean_int, time_short_int, v_mean_left_ind, v_mean_right_ind)
        vavg = compute_mean_velocity(v_mean_int, time_short_int, v_mean_left_ind, v_mean_right_ind)
        text = '$\overline{v^2_p} / \overline{v_p} = %.2f mm/s$ ' % np.abs(veff)
        graph.addtext(ax7, text=text, subplot=212, option='bl', fontsize=10)
        graph.labelaxes(ax7, 'Time [s]', 'Piston velocity $v_p$ [mm/s]')

        # Save a velocity profile in png
        plotfilename = os.path.join(fn_noext, 'plots/pos_vel_time_averaged_over%dperiods' % int(len(time)/frame_rate))
        graph.save(plotfilename, verbose=True)

        # Update a data dictionary for plot
        key = 'span' + '%04.1f'% span
        subkey = 'vp_commanded' + str(vp_commanded)
        update_data_dict(data_dict, key, subkey, np.abs(veff))

        key = 'span' + '%04.1f'% span
        subkey = 'vp_commanded' + str(vp_commanded)
        update_data_dict(vmin_dict, key, subkey, np.abs(np.min(v_mean_int)))

        #Save effective velocity in pkl
        filepath = os.path.join(fn_noext, 'plots/veff.pkl')
        rw.write_pickle(veff, filepath)
        filepath = os.path.join(fn_noext, 'plots/vavg.pkl')
        rw.write_pickle(vavg, filepath)
        filepath = os.path.join(fn_noext, 'plots/strokelength.pkl')
        rw.write_pickle(actual_span, filepath)

        plt.close('all')


    if args.datacomp:
        # Plot using a data dictionary
        for key in sorted(data_dict.keys(), reverse=True):
            # Specify keys you would not like to show on the plot
            #skipkeylist = ['span10.2', 'span06.4', 'span03.8']
            skipkeylist=[]
            if not key in skipkeylist:
                label = key + 'mm'
                vp_commanded_list, veff_list, vmin_list = [], [], []
                for subkey in data_dict[key]:
                    vp_commanded_list.append(float(subkey[12:]))
                    veff_list.append(data_dict[key][subkey])
                    vmin_list.append(vmin_dict[key][subkey])
                # Sort the data lists
                vp_commanded_list_for_veff, veff_list = fa.sort_two_arrays_using_order_of_first_array(vp_commanded_list, veff_list)
                vp_commanded_list_for_vmin, vmin_list = fa.sort_two_arrays_using_order_of_first_array(vp_commanded_list, vmin_list)
                # Plot veff and vmin vs commanded velocity for each stroke length
                fig7, ax8 = graph.plot(vp_commanded_list_for_veff, veff_list, fignum=7, marker='o', linestyle='-', label=label)
                fig8, ax9 = graph.plot(vp_commanded_list_for_vmin, vmin_list, fignum=8, marker='o', linestyle='-', label=label)

        plt.figure(7)
        plt.legend()
        graph.labelaxes(ax7, 'Commanded velocity [mm/s]', '|$\overline{v^2_p} / \overline{v_p}$| [mm/s]')
        graph.setaxes(ax7, 0, 1100, 0, 800)
        # Save a figure
        plotfilename = os.path.join(tracking_settings['input_dir'], 'motor_characteristics/motor_characteristics')
        graph.save(plotfilename, verbose=True)
        # Save data in pkl
        pickledir = os.path.join(tracking_settings['input_dir'], 'motor_characteristics/')
        pickle_rw.write(vp_commanded_list_for_veff, pickledir + 'vp_commanded_for_veff.pkl')
        pickle_rw.write(veff_list, pickledir + 'veff.pkl')


        plt.figure(8)
        plt.legend()
        graph.labelaxes(ax8, 'Commanded velocity [mm/s]', '|$v_{min}$| [mm/s]')
        graph.setaxes(ax8, 0, 1100, 0, 1100)
        # Save a figure
        plotfilename = os.path.join(tracking_settings['input_dir'], 'motor_characteristics/motor_characteristics_vmin')
        graph.save(plotfilename, verbose=True)
        # Save data in pkl
        pickledir = os.path.join(tracking_settings['input_dir'], 'motor_characteristics/')
        pickle_rw.write(vp_commanded_list_for_vmin, pickledir + 'vp_commanded_for_vmin.pkl')
        pickle_rw.write(vmin_list, pickledir + 'vmin.pkl')

    print 'Done'





