import cine
import h5py
import isolum_rainbow
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pylab as P
import sys
import time
import glob
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Wedge, Polygon, Rectangle
from subprocess import call
import lepm.dataio as dio
try:
    import Motor_tracking.movie_instance as mi
except:
    import motor_tracking.movie_instance as mi

from scipy.signal import savgol_filter

'''functions for plotting experiments and coverting to movie'''


def get_frame_points_and_time(data, keys, i):
    pts = []
    for key in keys:
        if key != 'times':
            pts.append(data[key][i])
    pts = np.array(pts)
    time = data['times'][i]

    return time, np.array(pts)


def get_average_positions(data, keys, range=[0, -1]):
    xavg = []
    yavg = []
    for key in keys:
        brk = key.split('_')[-1]
        if key != 'times':
            dt = np.array(data[key])
            dt = dt[range[0]:range[1]]

            xavg.append(np.mean(dt[:, 0]))
            yavg.append(np.mean(dt[:, 1]))

    return xavg, yavg


def apply_filter(data, keys):
    for key in keys:
        if key != 'times':
            single_data = np.array(data[key]).copy()

            single_data[:, 0] = savgol_filter(single_data[:, 0], 11, 1)
            single_data[:, 1] = savgol_filter(single_data[:, 1], 11, 1)
            try:
                new_dset = data.create_dataset(key + '_smoothed', np.shape(single_data), dtype='float', data=single_data)
            except:
                break
    return data


def plot_on_frame(frame, time, pts, avg, plotting_mag=1, save_name='frame.png', color_by='phase',
                  cmap='isolum_rainbow', plot_dict={'p_axT':False}, alpha=0.6, dpi=100, mark_pts=False, ptsz=20,
                  mark_line=True):
    """

    Parameters
    ----------
    frame
    time
    pts
    avg
    plotting_mag
    save_name
    color_by
    cmap
    plot_dict

    Returns
    -------

    """
    p_axT = plot_dict['p_axT']
    # print p_axT
    # frame = adjust_frame(frame, 0.2, 0.5)

    num_gyros = len(pts)

    x_for_t = pts[:, 0] - avg[:, 0]
    y_for_t = pts[:, 1] - avg[:, 1]
    patch = []
    bol = []
    colors = []

    for j in xrange(num_gyros):
        circ = Circle((avg[j, 0], avg[j, 1]), radius=plotting_mag * (np.sqrt(x_for_t[j]**2 + y_for_t[j]**2)))

        if color_by == 'amplitude':
            colors.append(np.sqrt((plotting_mag * x_for_t[j]) ** 2 + (plotting_mag * y_for_t[j]) ** 2))

        bol.append(True)
        patch.append(circ)

    if color_by == 'phase':
        # Plot patches colored by the phase of each gyro
        colors = (np.arctan2(y_for_t, x_for_t)) % (2 * np.pi)
        p = PatchCollection(patch, cmap=cmap, alpha=alpha, linewidth =0)
        p.set_array(np.array(colors))
        p.set_clim([0, 2 * np.pi])
    elif color_by == 'none':
        # Plot patches colored by the phase of each gyro
        colors = (np.arctan2(y_for_t, x_for_t)) % (2 * np.pi)
        p = PatchCollection(patch, cmap=cmap, alpha=alpha)
        p.set_array(np.array(colors))
        p.set_clim([0, 2 * np.pi])
    else:
        colors = np.array(colors)
        # colors = np.sqrt((plotting_mag* x_for_t) ** 2 + (plotting_mag*y_for_t) ** 2)
        p = PatchCollection(patch, cmap=cmap, alpha=alpha)
        p.set_array(np.array(colors))
        p.set_clim([0, 30])

    # x_for_t = avg[:, 0] + plotting_mag * x_for_t
    # y_for_t = avg[:, 1] + plotting_mag * y_for_t

    bol = np.array(bol, dtype=bool)
    # axes constructor axes([left, bottom, width, height])

    if not p_axT:
        fig = plt.figure(figsize=(5, 4.20))
        p_ax = P.axes([0.0, 0.0, 1., 1.])

        plt.imshow(frame, cmap=cm.Greys_r)
        # plt.show()

    else:
        p_ax = plot_dict['p_ax']
        plt.imshow(frame, cmap=cm.Greys_r, interpolation='none')

    p_ax.axes.get_xaxis().set_ticks([])
    p_ax.axes.get_yaxis().set_ticks([])
    p_ax.add_collection(p)
    # if color_by != 'none':
    # abc2 = p_ax.scatter(x_for_t[bol], y_for_t[bol], c='w', alpha=1)

    if not p_axT:
        bbox_props = dict(boxstyle="round", fc="k", ec="0.2", alpha=0)
        p_ax.text(25, 45, '$t = %0.1f s$' % time, ha="left", va="baseline", size=26, bbox=bbox_props, color='w',
                  family='sans-serif')
    # plt.xlim(100, 1000)
    # plt.ylim(580, 0)

    # If we are to plot the current position as scatterplot, do so here
    if mark_pts:
        p_ax.scatter(x_for_t * plotting_mag + avg[:, 0], y_for_t * plotting_mag + avg[:, 1], c='w', s=ptsz, zorder=9999)

    if mark_line:
        for j in range(num_gyros):
            xpos = x_for_t[j] * plotting_mag + avg[j, 0]
            ypos = y_for_t[j] * plotting_mag + avg[j, 1]
            p_ax.plot([avg[j, 0], xpos], [avg[j, 1], ypos], 'w-', lw=2, zorder=9998)

    # plt.show()
    # sys.exit()
    if not p_axT:
        plt.savefig(save_name, dpi=dpi)
        # if color_by != 'none':
        # abc2.remove()
        plt.close()


def make_frames(root_dir, video_path, overwrite=True, fn_for_movie=None, color_by='phase', cmap='isolum_rainbow',
                plot_mod=1, plot_mag=10, allow_ptsonly=False, neighborhood=50, rm_frames=True, dpi=100,
                min_bright=0.3, max_bright=0.8, fps=30, mark_pts=False, ptsz=20):
    """Create the frames of the cine with displacement circles overlaid, and convert into a movie

    Parameters
    ----------
    root_dir : str
        directory where the movie will go, where frames are output
    video_path : str
        the path of the cine file itself (full path, including the filename)
    fn_for_movie : str
        If None, grabs the simulation dir name and saves the movie with that name
    color_by :
    cmap :
    plot_mod :
    plot_mag : float or int
        scale factor for displacement circles overlaying cine frames
    neighborhood : int
        The window over which we average to get the average position of the gyro. If smaller, allows a moving
        equilibrium position

    Returns
    -------
    """
    save_directory = dio.prepdir(os.path.join(root_dir, 'movie_frames_' + color_by + '_filtered_linear' + cmap))
    print 'making the frames'
    mod_list = [16, 8, 4, 2, 1]

    # Make the name of the movie
    if fn_for_movie is None:
        nbrhoodstr = '_nbr' + str(neighborhood)
        ampstr = '_amp{0:0.1f}'.format(plot_mag).replace('.', 'p')
        fn_for_movie = save_directory.split('/')[-2] + nbrhoodstr + ampstr + '.mov'

    # print 'overwrite = ', overwrite
    # print 'looking for ', os.path.join(root_dir, fn_for_movie)
    # print 'glob ->', glob.glob(os.path.join(root_dir, fn_for_movie))
    # print 'glob = ', (not glob.glob(os.path.join(root_dir, fn_for_movie)))

    if overwrite or not glob.glob(os.path.join(root_dir, fn_for_movie)):
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)

        # Instantiate a movie
        print 'instantiating movie with video_path = ', video_path
        movie = mi.GyroMovie(video_path)
        movie.set_min_max_val(min_bright, max_bright)
        path = os.path.join(root_dir, 'com_data.hdf5')
        print 'loading the data...'
        data = h5py.File(path, 'r')
        keys = data.keys()
        length = len(data['times'])
        print 'loaded the data'

        # Check if there are all the frames already in the save_directory
        moviefiles = glob.glob(save_directory + '*.png')
        if len(moviefiles) > length - 2:
            # frames have already been written -- create movie out of them
            # Make the movie
            print 'creating movie: ', os.path.join(root_dir, fn_for_movie)
            # call(["ffmpeg", "-r", "50", "-start_number", str(0), "-i", save_directory + "/%05d.png",
            #       "-f", "mp4", "-pix_fmt", "yuv420p",  # save_directory + "h264",
            #       os.path.join(root_dir, fn_for_movie)])
            call(['./ffmpeg', '-r', "15", "-start_number", str(0),
                  '-i', save_directory + '/%05d.png', os.path.join(root_dir, fn_for_movie),
                  '-vcodec', 'libx264',
                  '-profile:v', 'main',
                  '-crf', '12', '-threads', '0', '-r', '100', '-pix_fmt', 'yuv420p'])

            if rm_frames:
                print 'calling rmdir ', save_directory
                call(['rm', '-r', save_directory])
        else:
            # if allow_ptsonly:
            #     try:
            #         c = cine.Cine(video_path)
            #     except:
            #         c = []
            # else:
            #     c = cine.Cine(video_path)

            # First go through the data quickly, then save finer resolution in time
            for j in xrange(len(mod_list)):
                avg_x = []
                avg_y = []
                plot_num = 0
                for i in xrange(length):
                    if i % plot_mod == 0:
                        print 'mnf: ', i
                        num = np.floor(i / plot_mod) + i % plot_mod

                        frame_name = os.path.join(save_directory, '%05d.png' % num)
                        # If the image file doesn't already exist, write it here
                        if i % mod_list[j] == 0 and not (os.path.isfile(frame_name)):
                            # form lower bound for average position
                            if i < neighborhood:
                                lb = 0
                            else:
                                lb = i - neighborhood

                            # form upper bound for average position
                            if length - plot_mod < i + neighborhood:
                                ub = length - plot_mod
                            else:
                                ub = i + neighborhood

                            xs, ys = get_average_positions(data, keys, range=[lb, ub])

                            avg_x.append(xs)
                            avg_y.append(ys)
                            time, pts = get_frame_points_and_time(data, keys, i)

                            if allow_ptsonly:
                                 try:
                                    # frame = c[i].astype('f')
                                    movie.extract_frame_data(i)
                                    movie.adjust_frame()
                                    frame = movie.current_frame

                                    print frame
                                 except:
                                    frame = np.zeros((600, 600))
                            else:
                                movie.extract_frame_data(i)
                                movie.adjust_frame()
                                frame = movie.current_frame

                            plot_on_frame(frame, time, pts, np.array([xs, ys]).T, save_name=frame_name,
                                          plotting_mag=plot_mag, color_by=color_by, cmap=cmap, dpi=dpi,
                                          mark_pts=mark_pts, ptsz=ptsz)
                            plot_num += 1

                        # print 'mnf: mod_list[j]=', mod_list[j]
                        # print 'i =', i
                        # print 'length - plot_mod - 1=', length - plot_mod - 1
                        if mod_list[j] == 1 and i >= length - plot_mod - 1:
                            # Make the movie
                            print 'creating movie: ', os.path.join(root_dir, fn_for_movie)
                            # call(["ffmpeg", "-r", "50", "-start_number", str(0), "-i", save_directory + "/%05d.png",
                            #       "-f", "mp4", "-pix_fmt", "yuv420p",  # save_directory + "h264",
                            #       os.path.join(root_dir, fn_for_movie)])
                            call(['./ffmpeg', '-r', str(fps), "-start_number", str(0),
                                  '-i', save_directory + '/%05d.png', os.path.join(root_dir, fn_for_movie),
                                  '-vcodec', 'libx264',
                                  '-profile:v', 'main',
                                  '-crf', '12', '-threads', '0', '-pix_fmt', 'yuv420p'])

                            if rm_frames:
                                print 'calling rmdir ', save_directory
                                call(['rm', '-r', save_directory])

                            # except:
                            #     print 'Could not make movie, skipping...'
                        #     # call(
                        #     #   ["ffmpeg", "-r", "50", "-i", "video.mp4", output_dir + final_fn_for_movie + '.mp4'])


if __name__ == '__main__':
    video_path = '/Volumes/labshared2/Lisa/2017_02_21/7p_0p0A_5p5A_1.cine'
    # '#'/Users/lisa/Dropbox/Research/2017_02_17_data/untracked2/2p00hz_0p0amps_2.cine'
    #  '/Users/lisa/Dropbox/Research/2017_02_17_data/untracked2/2p00hz_0p0_5p0_ramp.cine'
    root_dir = '/Volumes/labshared2/Lisa/2017_02_21/tracked/7p_0p0A_5p5A_1/'
    # '/Users/lisa/Dropbox/Research/2017_02_17_data/tracked/2p00hz_0p0_5p0_ramp_2/'
    # ffmpeg -r 120 -start_number 0 -i %05d.png -f mp4 h264 -pix_fmt yuv420p video_s.mp4
