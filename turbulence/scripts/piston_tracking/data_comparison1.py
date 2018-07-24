"""
Comparison of tracking results between software outputs and tracking algorithm (Hough algorithm) outputs
"""
import numpy as np
import library.display.graph as graph
import library.tools.rw_data as rw
import library.basics.formatstring as fs
import library.basics.formatarray as fa
import glob
import sys
import os
import argparse
import h5py
#import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import integrate
import matplotlib.pyplot as plt
import library.tools.process_data as process


base_dir = '/Volumes/labshared3-1/takumi/Similar_flows/'
#base_dir = '/Volumes/labshared3-1/takumi/Similar_flows/pistontuning/tracking_comparison'
# data_tracking_basedir_list = [base_dir + 'pistontuning/dx0.0125autophased_polarityfixed2',
#                               base_dir + 'pistontuning/dx0.0125autophased_polarityfixed2_piston2']
data_tracking_basedir_list = [base_dir + 'pistontuning/tracking_comparison']

piston_res = 51.2 / 4096  # 0.0125mm

parser = argparse.ArgumentParser(description='Comparison between tracking results and CME2 software outputs')
parser.add_argument('-fx', '--fx', help='Conversion factor: ', type=float, default=1.)
parser.add_argument('-start', '--start', help='frames used to analyze [start, total_frame_number - end]', type=int, default=0)
parser.add_argument('-end', '--end', help='frames used to analyze (start, total_frame_number - end]', type=int, default=0)
parser.add_argument('-shiftper', '--shiftper', help='Shift the data arrays by this much percentage forward when plotting', type=float, default=0.2)
args = parser.parse_args()


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

    # fig = plt.figure()
    # plt.scatter(compare[:, 1], compare[:, 2], c=range(len(compare)), cmap=plt.cm.coolwarm)

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
        # plt.gca().set_aspect(1)
        # plt.savefig(os.path.join(root_dir, 'color_by_number.png'))

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

def get_velocity_extrema_near_lift(velocity, plot=False):
    """
    Returns indices of array where the sign of elements change near the minimum value of the array
    Parameters
    ----------
    velocity
   a plot

    Returns
    -------

    """
    # Find a minimum value and its index
    v_min_ind, v_min = np.argmin(velocity), np.amin(velocity)
    # Split an array into two parts using the minimum value
    v1, v2 = velocity[:v_min_ind], velocity[v_min_ind:]
    # Detect a sign flip of the left array
    signflip_indices = fa.detect_sign_flip(v1)
    v_mean_left_ind = signflip_indices[-1]
    # Detect a sign flip of the right array
    signflip_indices = fa.detect_sign_flip(v2)
    v_mean_right_ind = len(v1) + signflip_indices[0]

    return v_mean_left_ind, v_mean_right_ind


for data_tracking_basedir in data_tracking_basedir_list:
    if data_tracking_basedir == base_dir + 'pistontuning/dx0.0125autophased_polarityfixed2':
        fx = 0.1744 #mm/px
    if data_tracking_basedir == base_dir + 'pistontuning/dx0.0125autophased_polarityfixed2_piston2':
        fx = 0.1599 #mm/px
    if data_tracking_basedir == base_dir + 'pistontuning/no_movement_test':
        fx = 0.1581 #mm/px
    if data_tracking_basedir == base_dir + 'pistontuning/tracking_comparison':
        fx = 0.1380 #mm/px

    csv_files = glob.glob(data_tracking_basedir + '/*.csv')
    print csv_files
    for csv_file in csv_files:
        print csv_file

        # Load csv data generated by CME2 software
        csv_data = rw.read_data(csv_file, delimiter=',')
        # Format the data array: delete the first row and the forth column
        csv_data = np.delete(csv_data, obj=0, axis=0)
        csv_data = np.delete(csv_data, obj=4, axis=1)
        time_csv = csv_data[:, 0]
        commanded_pos_csv = -(csv_data[:, 1] - np.max(csv_data[:, 1])) * piston_res
        commanded_vel_csv = np.gradient(commanded_pos_csv, time_csv)
        pos_csv = -(csv_data[:, 2] - np.max(csv_data[:, 2])) * piston_res
        vel_csv = -csv_data[:, 3]
        time_v_csv, vel_csv = process.compute_velocity_simple(time_csv, pos_csv)
        # Load hdf5 data generated by piston_tracking.py
        #
        data_tracking_dir, ext = os.path.splitext(csv_file)
        print data_tracking_dir, ext
        data_tracking_dir_head, data_tracking_dir_tail = os.path.split(data_tracking_dir)
        print data_tracking_basedir
        print data_tracking_dir
        print data_tracking_dir_tail

        print '_______'
        file = data_tracking_basedir + '/' + data_tracking_dir_tail + '/steps/steps.hdf5'
        print file
        #time_hdf5, x_pos_hdf5, y_pos_hdf5 = get_position_from_step_hdf5(data_tracking_dir)
        time_hdf5, y_pos_hdf5, x_pos_hdf5 = get_position_from_step_hdf5(data_tracking_dir)



        # read encoder output
        datafilepath_enc = data_tracking_basedir + '/encoder/' + data_tracking_dir_tail + '.txt'
        piston_data = np.loadtxt(datafilepath_enc)
        time_enc, y_pos_enc = piston_data[0], piston_data[1]
        time_enc, y_pos_enc = np.array(time_enc[000:100]), np.array(y_pos_enc[00:100])
        time_enc = time_enc - np.min(time_enc) + 0.267
        y_pos_enc = y_pos_enc - np.min(y_pos_enc)

        time_v_enc, vel_enc = process.compute_velocity_simple(time_enc, y_pos_enc)
        time_v_enc = time_v_enc - 0.002
        print len(y_pos_enc)
        # plt.plot(time_enc, y_pos_enc)
        # plt.show()



        x_pos_hdf5[:] = (x_pos_hdf5[:] - np.min(x_pos_hdf5)) * args.fx
        y_pos_hdf5[:] = (y_pos_hdf5[:] - np.min(y_pos_hdf5)) * args.fx

        # Get experimental parameters from file name
        frame_rate = fs.get_float_from_str(data_tracking_dir_tail, 'fps', '_D')
        span = fs.get_float_from_str(data_tracking_dir_tail, 'piston', 'mm_v')
        vp_commanded = fs.get_float_from_str(data_tracking_dir_tail, '_v', 'mms')
        freq = fs.get_float_from_str(data_tracking_dir_tail, 'freq', 'Hz')

        # # Compute velocity from position (hdf5)
        u_hdf5 = np.gradient(x_pos_hdf5) * frame_rate
        v_hdf5 = np.gradient(y_pos_hdf5) * frame_rate

        # Delete some elements from arrays if commanded on terminal
        if not args.end==0:
            x_pos_hdf5 = x_pos_hdf5[args.start:-args.end]
            y_pos_hdf5 = y_pos_hdf5[args.start:-args.end]
            u_hdf5 = u_hdf5[args.start:-args.end]
            v_hdf5 = v_hdf5[args.start:-args.end]
            #print 'Number of frames in cine: %d' % len(time_hdf5)
            time_hdf5 = time_hdf5[args.start:-args.end]
        else:
            x_pos_hdf5 = x_pos_hdf5[args.start:]
            y_pos_hdf5 = y_pos_hdf5[args.start:]
            u_hdf5 = u_hdf5[args.start:]
            v_hdf5 = v_hdf5[args.start:]
            #print 'Number of frames in cine: %d' % len(time_hdf5)
            time_hdf5 = time_hdf5[args.start:]

        ## Roll y_pos_hdf5 data
        x_pos_hdf5 = np.roll(x_pos_hdf5, int(len(x_pos_hdf5) * args.shiftper))
        # plt.plot(time_hdf5, x_pos_hdf5)
        # plt.show()

        # Compute velocity from position
        u_hdf5 = np.gradient(x_pos_hdf5) * frame_rate
        v_hdf5 = np.gradient(y_pos_hdf5) * frame_rate

        # This data set has a peculiar value, so fix it
        if data_tracking_dir_tail == '20180405_4000fps_span5p0mm_v50mms_f1Hz_setting2':
            v_hdf5 = process.clean_multi_dim_array(v_hdf5, cutoff=150)
        if data_tracking_dir_tail == '20180405_4000fps_span1p9mm_v50mms_f1Hz_setting2':
            v_hdf5 = process.clean_multi_dim_array(v_hdf5, cutoff=100)

        ## Let time start at 0.
        time_hdf5 = time_hdf5 - np.min(time_hdf5)

        # Align the software results and tracking results
        # Get where velocity curve flips a sign near the minimum value
        u_hdf5_left_ind, u_hdf5_right_ind = get_velocity_extrema_near_lift(u_hdf5)
        vel_csv_left_ind, vel_csv_right_ind = get_velocity_extrema_near_lift(vel_csv)
        # # Use the LEFT value to compute how many indices the array has to be shifted
        shift_ind, shift_val = fa.find_nearest(time_hdf5, time_csv[vel_csv_left_ind])
        shift_arg = x_pos_hdf5.shape[0] - u_hdf5_left_ind + shift_ind


        #### konkaidake
        shift_sine_hdf5_arg, xmax_hdf5 = fa.find_max(x_pos_hdf5)
        shift_sine_csv_arg, ymax_csv = fa.find_max(pos_csv)



        # Alternatively, use the RIGHT value to compute how many indices the array has to be shifted
        shift_ind, shift_val = fa.find_nearest(time_hdf5, time_csv[vel_csv_right_ind])
        shift_arg = x_pos_hdf5.shape[0] - u_hdf5_right_ind + shift_ind

        #### konkaidake
        shift_ind, shift_val = fa.find_nearest(time_hdf5, time_csv[shift_sine_csv_arg])
        shift_ind = shift_ind + 80
        shift_arg = x_pos_hdf5.shape[0] - shift_sine_hdf5_arg + shift_ind


        # Match time of the two data curves (tracking code results and software results) at the peak
        x_pos_hdf5 = np.roll(x_pos_hdf5, shift_arg)
        u_hdf5 = np.roll(u_hdf5, shift_arg)




        time_hdf5 = time_hdf5 - 0.127

        # Calculate actual stroke length
        ## Get a small numpy array from t=0 to t=tmax
        tmin, tmax = np.min(time_csv), np.max(time_csv)
        short_time_hdf_max_ind, short_time_hdf_max =fa.find_nearest(time_hdf5, tmax)
        short_time_hdf5 = time_hdf5[0:short_time_hdf_max_ind]
        short_x_pos_hdf5 = x_pos_hdf5[0:short_time_hdf_max_ind]
        short_pos_csv = pos_csv[0:short_time_hdf_max_ind]
        stroke_length_hdf5 = np.max(short_x_pos_hdf5)-np.min(short_x_pos_hdf5)
        stroke_length_csv = np.max(short_pos_csv)-np.min(short_pos_csv)

        # Position Plot
        # fig1, ax1 = graph.plot(time_csv, commanded_pos_csv, label='Commanded position (software outputs)', fignum=1, subplot=211, figsize=(12, 8))
        fig1, ax1 = graph.plot(time_csv, pos_csv, label='Actual position (software outputs)', color='C1', fignum=1, subplot=211, figsize=(12, 8), alpha = 0.5)
        fig1, ax1 = graph.plot(time_hdf5, x_pos_hdf5, label='Actual position (tracking result)',color='C2',  fignum=1, subplot=211, figsize=(12, 8))
        #fig1, ax1, color_patch1 = graph.errorfill(time_hdf5, y_pos_hdf5, fx, label='Actual position (tracking result)', fignum=1, subplot=211,
                                #figsize=(12, 8), color='C2')
        fig1, ax1 = graph.plot(time_enc, y_pos_enc, label='Actual position (encoder result)', fignum=1, subplot=211,
                               figsize=(12, 8), color='C3', alpha = 0.5)

        # Velocity Plot
        # fig1, ax2 = graph.plot(time_csv, commanded_vel_csv, label='Commanded velocity (software outputs)', fignum=1, subplot=212,
        #                        figsize=(12, 8))
        fig1, ax2 = graph.plot(time_csv, vel_csv, label='Actual velocity (software outputs)',color='C1', fignum=1, subplot=212, figsize=(12, 8), alpha = 0.5)
        fig1, ax2 = graph.plot(time_hdf5, u_hdf5, label='Actual velocity (tracking result)',color='C2', fignum=1, subplot=212, figsize=(12, 8), alpha = 0.5)
        fig1, ax2 = graph.plot(time_v_enc, vel_enc, label='Actual position (encoder result)', fignum=1, subplot=212,
                               figsize=(12, 8), color='C3', alpha = 0.5)
        # graph.setaxes(ax1, tmin, tmax, -0.2, span * 1.3)
        # graph.setaxes(ax2, tmin, tmax, -500, 100)
        graph.legend(ax1, loc=1)
        graph.legend(ax2, loc=4)
        graph.addtext(ax1, 'Actual stroke length(tracking code): %.2f + %.2f mm' % (stroke_length_hdf5, 2*fx), option='bl', fontsize=10)
        graph.addtext(ax1, 'Actual stroke length(software): %.2f mm' % stroke_length_csv, option='bl2', fontsize=10, alpha = 0.5)
        graph.labelaxes(ax1,'Time [s]', 'Position y [mm]')
        graph.labelaxes(ax2, 'Time [s]', 'Velocity [mm/s]')
        graph.suptitle(data_tracking_dir_tail)
        graph.setaxes(ax1, 0.15,0.8,-0.5,13)
        graph.setaxes(ax2, 0.15, 0.8, -550, 200)
        # Draw a line where the two data were aligned
        # graph.axvline(ax1, x=time_csv[vel_csv_right_ind], linestyle='--', color='k')
        # graph.axvline(ax2, x=time_csv[vel_csv_right_ind], linestyle='--', color='k')


        # Scatter plot where the two data (tracking code results and software outputs) were aligned
        # if v_hdf5_left_ind+shift_arg >= time_hdf5.shape[0]:
        #     fig1, ax1 = graph.scatter([time_hdf5[v_hdf5_right_ind + shift_arg - time_hdf5.shape[0]]], [v_hdf5[v_hdf5_right_ind + shift_arg - time_hdf5.shape[0]]], fignum=1, subplot=212,
        #                   marker='x', color='C2', figsize=(12,8))
        # else:
        #     fig1, ax1 = graph.scatter(time_hdf5[v_hdf5_right_ind+shift_arg], v_hdf5[v_hdf5_right_ind+shift_arg], fignum=1, subplot=212,
        #                   marker='x', color='C2', figsize=(12,8))
        # fig1, ax1 = graph.scatter(time_csv[vel_csv_right_ind], vel_csv[vel_csv_right_ind], fignum=1, subplot=212,
        #                   marker='x', color='C1', figsize=(12,8))

        plot_filename = '/comparison/' + data_tracking_dir_tail
        graph.save(data_tracking_basedir + plot_filename, verbose=True)
        plt.close('all')
        graph.show()

