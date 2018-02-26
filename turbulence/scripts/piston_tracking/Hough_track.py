import h5py
import link_points as lp
import matplotlib.image as mpimg
import movie_instance as mi
import numpy as np
import os
import sys
import time
import tracking_helper_functions as thf
import Experiment_movies
from settings import tracking_settings
import matplotlib.pyplot as plt
# from click_pts import ImagePoint
from lepm.build.roipoly import RoiPoly
import glob
import cPickle as pkl

''''''


def run_tracking(tracking_settings, reverse=False, overwrite=False):
    """

    Parameters
    ----------
    tracking_settings : dict
        tracking_settings contains key, val pairs:
            'input_dir': input_dir,
            'output_dir': input_dir,
            'first_frame': 0,
            'last_frame': -1,
            'pix': 12,
            'min_max_radius': list of two floats or ints
                the minimum and maximum radius of circles for which to search in Hough transform
            'cf': [0],
            'min_max_val': list of two floats
                the lower and upper limits on the brightness before Hough transform

    Returns
    -------

    """
    fns = []

    if 'input_dir' in tracking_settings:
        fn = tracking_settings['input_dir']

        spl = fn.split('.')[-1]

        if spl == 'cine':
            fns = [fn]
        else:
            # find all cines in directory
            print 'searching for cines in input_dir = ', fn
            fns = thf.find_files_by_extension(fn, '.cine', tot=True)

            if len(fns) < 1:
                fns = [fn]

    else:
        print 'No input file selected.  Exiting'
        sys.exit()

    output_dir = thf.set_output_directory()

    outputs = []
    if reverse:
        fns.reverse()

    print 'fns = ', fns

    for fn in fns:
        # Look at each file, create a movie instance
        print 'examining fn = ', fn
        print 'extracting frame_rate from file name...'
        frame_rate = extract_frame_rate(fn)
        movie = mi.GyroMovie(fn, frame_rate=frame_rate)
        output = fn.split('/')[-1]
        output = output_dir + '/' + output.split('.')[0]
        outputs.append(output)
        if not os.path.exists(output):
            os.mkdir(output)
        else:
            output = output
            if not os.path.exists(output):
                os.mkdir(output)

        # Check variance or variation of the frames
        # movie.set_maxdiff_frame()
        # movie.save_maxdiff_frame(name=output + '/' + 'maxdiffframe.png')
        #
        # movie.set_variance_frame()
        # movie.save_variance_frame(name=output + '/' + 'varframe.png')

        if not os.path.exists(os.path.join(output, 'com_data.hdf5')) or overwrite:
            # Dump tracking settings in this output dir
            with open(os.path.join(output, 'tracking_settings.pkl'), "wb") as tsfilename:
                pkl.dump(tracking_settings, tsfilename)

            # Make output dir for moniroting tracking progress
            checks = os.path.join(output, 'checks')
            if not os.path.exists(checks):
                os.mkdir(checks)

            # for saving steps
            path_to_step_data = os.path.join(output, 'steps')
            if not os.path.exists(path_to_step_data):
                os.mkdir(path_to_step_data)

            if 'pix' in tracking_settings:
                movie.set_tracking_size(tracking_settings['pix'])
            else:
                # Default value is 10
                movie.set_tracking_size(10)

            if 'min_dist' in tracking_settings:
                movie.min_dist = tracking_settings['min_dist']

            if 'min_max_radius' in tracking_settings:
                movie.min_radius = tracking_settings['min_max_radius'][0]
                movie.max_radius = tracking_settings['min_max_radius'][1]

            if 'centroid_clipminmax' in tracking_settings:
                movie.set_centroid_clip(tracking_settings['centroid_clipminmax'][0],
                                        tracking_settings['centroid_clipminmax'][1])

            if 'min_max_val' in tracking_settings:
                movie.set_min_max_val(tracking_settings['min_max_val'][0], tracking_settings['min_max_val'][1])

            if 'first_frame' in tracking_settings:
                ff = tracking_settings['first_frame']
            else:
                ff = 0

            if 'last_frame' in tracking_settings:
                if tracking_settings['last_frame'] >= ff:
                    lf = tracking_settings['last_frame']
                else:
                    lf = movie.num_frames
            else:
                lf = movie.num_frames

            ######################################################################
            # Prepare ROI
            # First extract region of interest (roi)
            movie.extract_frame_data(ff)
            movie.adjust_frame()
            roifn = os.path.join(output, '') + 'roi.txt'
            if 'roi' in tracking_settings:
                # roi is either None (no roi), a numpy array (the roi), or 'click' (which tells hough_track to
                # define it here)
                if isinstance(tracking_settings['roi'], str):
                    if tracking_settings['roi'] in ['None', 'none']:
                        tracking_settings['roi'] = None
                    else:
                        # attempt to load the roi from a saved txt file
                        if glob.glob(roifn):
                            roi = np.loadtxt(roifn)
                        else:
                            # click on the points to define the roi
                            plt.imshow(movie.current_frame)
                            ax = plt.gca()
                            roi = RoiPoly(ax=ax, roicolor='r')
                            roi = np.dstack((roi.allxpoints, roi.allypoints))[0]
                            plt.close('all')

                        tracking_settings['roi'] = roi
                elif isinstance(tracking_settings['roi'], list):
                    tracking_settings['roi'] = np.array(roi)

                # Save roi in output dir and in movie instance if not None
                movie.roi = tracking_settings['roi']
                roi = tracking_settings['roi']
                if roi is not None:
                    np.savetxt(roifn, roi)
            else:
                tracking_settings['roi'] = None
            ######################################################################

            # Decide if we use difference between frames as a method to track
            if 'diff' in tracking_settings:
                if tracking_settings['diff']:
                    trackdiff = True
                else:
                    trackdiff = False
            else:
                trackdiff = False

            ######################################################################
            # Get num_times for how many times to recenter on centroid (default=1)
            if 'cent_num_times' in tracking_settings:
                num_times = tracking_settings['cent_num_times']
            else:
                num_times = 1

            ######################################################################
            # Subtract out the mean intensity in the image, if that is called for in track_settings
            if 'subtract_mean' in tracking_settings:
                subtract_mean = tracking_settings['subtract_mean']
                if subtract_mean:
                    movie.set_average_frame(first_ind=0, last_ind=None)
                    movie.adjust_avgframe()
            else:
                subtract_mean = False
            ######################################################################
            # Mutliply the image by a map that normalizes the background luminosity
            if 'normalize_background' in tracking_settings:
                normbg = tracking_settings['normalize_background']
                if normbg > 0:
                    movie.set_normfactor_frame(normbg, first_ind=0, last_ind=None)
                    normalize_background = True
                else:
                    normalize_background = False
            else:
                normalize_background = False
            ######################################################################
            if 'mask' in tracking_settings:
                if tracking_settings['mask']:
                    mask = True
                else:
                    mask = False
            else:
                mask = False

            ######################################################################
            if 'cf_reference_pts' in tracking_settings:
                if tracking_settings['cf_reference_pts'] in [None, 'cf_points']:
                    click_reference_pts = False
                else:
                    click_reference_pts = True
            else:
                click_reference_pts = False

            ######################################################################
            # Determine mask method (only used if mask is True)
            if 'mask_method' in tracking_settings:
                mask_method = tracking_settings['mask_method']
            else:
                mask_method = 'cf_coords'

            ######################################################################
            # Determine reference mask diameter (or radius?), only used if mask_method == 'cf_coords' and mask==True
            if 'refmask_thresh' in tracking_settings:
                refmask_thresh = tracking_settings['refmask_thresh']
            else:
                refmask_thresh = 16
            ######################################################################

            st = time.time()
            com_data = []
            for ii in xrange(lf - ff):
                # For the new current frame, get the data and adjust brightness
                ind = ii + ff
                movie.extract_frame_data(ind)
                movie.adjust_frame()

                if (ii in tracking_settings['cf']) or ('all' in tracking_settings['cf']):
                    # For cf frames (ie those in the list of tracking_settings['cf']), use special centroid-finding
                    # method to get the positions of the particles
                    # Note that a different centroid window size is used than for non-cf frames
                    if 'cf_method' in tracking_settings:
                        if tracking_settings['cf_method'] == 'hough':
                            print 'hough_track.py: taking hough transform...'
                            movie.find_points_hough()
                        if tracking_settings['cf_method'] == 'houghcentroid':
                            print 'hough_track.py: taking hough transform...'
                            movie.find_points_hough()
                            movie.center_on_bright(num_times=2, pix=tracking_settings['pix_cf'])
                        elif tracking_settings['cf_method'] == 'convolve':
                            print 'hough_track.py: convolving...'
                            movie.find_points_convolution(image_kernel_path=tracking_settings['convolution_image'],
                                                          roi=tracking_settings['roi'])
                        elif tracking_settings['cf_method'] == 'convolveclick':
                            print 'hough_track.py: convolving...'
                            movie.find_points_convolution(image_kernel_path=tracking_settings['convolution_image'],
                                                          roi=tracking_settings['roi'])
                        elif tracking_settings['cf_method'] == 'click':
                            print "hough_track.py: cf_method is 'click'..."
                            clickfn = os.path.join(output, 'click_pts.txt')
                            fns = glob.glob(clickfn)
                            plt.close('all')
                            if fns:
                                movie.frame_current_points = np.loadtxt(clickfn)
                            else:
                                points = movie.find_points_click(tool=tracking_settings['cf_rough_method'])
                                # save the points that have been clicked
                                header = 'locations of clicked original white dots at center of each gyro'
                                np.savetxt(clickfn, points, header=header)
                    else:
                        movie.find_points_hough()
                        # movie.find_points_convolution()

                    movie.save_frame_with_boxes(name=output + '/' + 'cfframe_%03d' % ind)
                    pix = tracking_settings['pix_cf']
                elif 'none' in tracking_settings['cf']:
                    # Manually click on every point in each frame
                    movie.find_points_click()
                    movie.save_frame_with_boxes(name=output + '/' + '%03d' % ind)
                    pix = None
                else:
                    pix = None

                # Store hough or convolution centers if ii == 0
                if ii == 0 and mask:
                    print 'ht: this is first frame and mask is true. This means that we have to load or select ' \
                          'reference points with which to do the mask'
                    print 'ht: Loading or saving reference points from/to txt file...'
                    # Create the mask to apply to all frames
                    # The reference points set the centers of the mask circles.
                    # First discern reference points (typically centers of displacements or place to put plotted
                    # excitation)
                    refptfn = os.path.join(output, 'reference_points.txt')
                    # Load or define clicked reference points
                    if glob.glob(refptfn):
                        movie.reference_points = np.loadtxt(refptfn)
                    else:
                        # If tracking_points['cf_reference_pts'] is not None or 'cf_points', click them individually
                        if click_reference_pts:
                            refpts = movie.click_reference_pts()
                        else:
                            refpts = movie.update_reference_pts()

                        # Save the reference points, which are typically positions for initially setting
                        # the centers of motion about which displacements are measured.
                        # The reference points set the centers of the mask circles.
                        print 'saving ' + refptfn
                        np.savetxt(refptfn, refpts, header='reference points for movie and mask')

                    print 'ht: Setting reference mask...'
                    if mask_method is None:
                        raise RuntimeError('mask is True, but mask_method is None!')

                    # Define output filename for loading/dumping
                    maskfn = os.path.join(output, 'reference_mask.txt')
                    if 'click' in mask_method and glob.glob(maskfn):
                        print 'Loading the reference mask since mask_method is click and a file is saved...'
                        refmask = np.loadtxt(maskfn)
                        movie.set_reference_mask(mask=refmask)
                    else:
                        movie.set_reference_mask(mask=None, mask_method=mask_method, thresh=refmask_thresh)
                        # Save the rois in the reference mask
                        np.savetxt(maskfn, movie.reference_mask, header='reference_mask')
                        plt.close('all')
                        plt.imshow(movie.reference_mask)
                        plt.savefig(os.path.join(output, 'reference_mask.png'))

                # Set the brightest pixels to zero if tracking_settings['kill_bright_thres'] > 0
                if tracking_settings['kill_bright_thres'] > 0:
                    print 'ht: Killing brightest pixels...'
                    thres = tracking_settings['kill_bright_thres'] * np.max(movie.current_frame.ravel())
                    killinds = movie.current_frame > thres
                    movie.current_frame[killinds] = 0

                # Update positions of the particles to be centroid of nearby brightness
                if trackdiff:
                    movie.center_on_bright_difference(num_times=num_times)
                    movie.update_previous_frame()
                else:
                    if subtract_mean:
                        movie.subtract_avgframe()
                    if normalize_background:
                        movie.adjust_normbg()
                    if ii > 0:
                        movie.mask_current_frame()
                    movie.center_on_bright(num_times=num_times, pix=pix)

                if 'tracked_image' in tracking_settings:
                    movie.save_frame_with_boxes(name=output + '/' + '%03d' % ind)

                if ii % tracking_settings['check_modii'] == 0:
                    et = time.time()
                    print 'frame', ii, 'tracked... ... %0.2f s per frame' % ((et - st) / (ii + 1))
                    movie.save_frame_with_boxes(name=checks + '/' + '%03d' % ind)

                if ii in tracking_settings['cf']:  # or ('all' in tracking_settings['cf']):
                    movie.save_frame(name=output + '/' + '%03d_nb' % ind)

                if ii == (lf - ff) - 1:
                    movie.save_frame_with_boxes(name=output + '/' + '%03d' % ind)

                timepts = np.array([[movie.current_time, movie.frame_current_points[jj, 0],
                                     movie.frame_current_points[jj, 1]]
                                    for jj in range(len(movie.frame_current_points))])

                com_data.append(timepts)
                path = os.path.join(path_to_step_data, 'steps.hdf5')

                if ii == 0:
                    ffile = h5py.File(path, "w")

                dset = ffile.create_dataset(('step_%05d' % ind), np.shape(timepts), dtype='float', data=timepts)

            ffile.close()

            lp.link_points(output)

    return fns, outputs


def extract_frame_rate(fn):
    """Pick out the frame rate from the file name

    Parameters
    ----------
    fn : str
        file name with '_175fps' or '_175pps' in it for 175 Hz framerate

    Returns
    -------
    frame_rate : float
        The frame rate as discerned by the file name
    """
    print 'fn = ', fn
    if 'fps' in fn:
        frame_rate = float(fn.split('fps')[0].split('_')[-1])
    elif 'pps' in fn:
        frame_rate = float(fn.split('pps')[0].split('_')[-1])
    else:
        raise RuntimeError('No frame rate in the filename. Cine framerate discernment is unreliable. Exiting.')

    return frame_rate
