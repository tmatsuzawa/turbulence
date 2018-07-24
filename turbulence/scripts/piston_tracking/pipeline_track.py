import Hough_track as ht
#import turbulence.scripts.piston_tracking.experiment_movies.experiment_movie as mnf

import sys
sys.path.append('/Users/stephane/Documents/git/takumi/turbulence/turbulence/scripts/piston_tracking/')
import experiment_movies.experiment_movie as mnf
import numpy as np
import time
from mode_drawing_functions.get_speed_and_plot import get_speed_and_plot_withresolution
import argparse
import mode_drawing_functions.mode_movie as modemov
import mode_drawing_functions.new_mode_functions as nmf


parser = argparse.ArgumentParser(description='Track videos of gyros.')
parser.add_argument('-check', '--check', help='Display intermediate results', action='store_true')
parser.add_argument('-overwrite', '--overwrite', help='Overwrite previous tracking results', action='store_true')
parser.add_argument('-reverse', '--reverse', help='Track videos in reverse order that they appear in dir',
                    action='store_true')
parser.add_argument('-speeds', '--speeds', help='Get motor spinning or precession speeds', action='store_true')
parser.add_argument('-movie', '--movie', help='Make a movie of the displacements over time', action='store_true')
parser.add_argument('-modes', '--modes', help='Make a movie of the normal modes', action='store_true')

# Options for speeds
parser.add_argument('-lospeed', '--low_cutoff_speed', help='Lower cutoff for speeds in fourier decomp',
                    type=float, default=100)
parser.add_argument('-hispeed', '--high_cutoff_speed', help='Upper cutoff for speeds in fourier decomp',
                    type=float, default=325)
# Options for modes
parser.add_argument('-amp', '--amplitude', help='Amplitude of mode movies',
                    type=float, default=2.)
parser.add_argument('-overwrite_modes', '--overwrite_modes', help='Overwrite the stills for movie of the normal modes',
                    action='store_true')


args = parser.parse_args()

if __name__ == '__main__':
    print 'running pipeline...'
    start_time = time.time()
    end_time = time.time()
    from settings import tracking_settings

    print 'tracking...'
    fns, outputs = ht.run_tracking(tracking_settings, reverse=args.reverse, overwrite=args.overwrite)
    print 'done tracking...'

    todo = range(len(outputs))
    if args.reverse:
        todo.reverse()

    for i in todo:
        fn_for_movie = outputs[i].split('/')[-1] + '.mp4'
        print fn_for_movie
        if args.speeds:
            coords, maxes, resolution = get_speed_and_plot_withresolution(fns[i][0:-5] + '/',
                                                                          lowcutoff=args.low_cutoff_speed,
                                                                          hicutoff=args.high_cutoff_speed,
                                                                          check=args.check)
            # speed_from_brightness_regions()
            print 'mean(speed) = ', np.mean(maxes)

        if args.movie:
            print 'outputs[i] = ', outputs[i]
            print 'fns[i] = ', fns[i]
            mnf.make_frames(outputs[i], fns[i], color_by='phase', cmap='isolum_rainbow', plot_mod=5, plot_mag=10,
                            allow_ptsonly=False, fn_for_movie=None, neighborhood=25, rm_frames=True, overwrite=False)

    if args.modes:
        from settings import input_dir
        nmf.save_mode_data(input_dir, thres=0.00001, overwrite=False)  # , freqmin=1.5, freqmax=2.5)
        modemov.make_mode_movie(input_dir, amp=args.amplitude, overwrite=args.overwrite_modes, percent_max=0.005)


