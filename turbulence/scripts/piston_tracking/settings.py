input_dir = '/Volumes/labshared3/noah/turbulence/midsize_box_motor_calibration/'
rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/'

# Notes
# roi is for selecting initial identification of particles only
#   it is not a mask applied to each frame (reference_mask)
# reference_mask is a mask applied to each frame before identification
#   this is useful for cropping out stationary bright spots that are sometimes near the gyro center

# many-gyro with coils settings, for displacement videos (for example, freqsweep or pulsepacket)
tracking_settings = {
    'input_dir': input_dir,
    'output_dir': input_dir,
    'first_frame': 0,
    'last_frame': -1,
    # how large of a centroid-finding window to use for non-cf frames
    'pix': 30,
    # pix_cf designates how large of a centroid-finding window to use for cf frames. Others use pix for window
    'pix_cf': 30,
    # Minimum distance between two particles to allow in Hough transform
    'min_dist': 50,
    # Bounds for hough transform
    'min_max_radius': [20, 50],
    # 'tracked_image': -1,
    # frames to identify particles with Hough Transform --> list of ints or ['all'] or ['none']
    'cf': [0],
    # Methods for cf (finding centers of each miniframe) are:
    # 1 'click' (manually click on each center),
    # 2 'convolve' (convolve an image to find matches), and
    # 3 'hough' (take hough transform to get centers of circles)
    'cf_method': 'click',  # 'click',  # 'convolve',  # 'hough',
    'cf_rough_method': 'click',
    'convolution_image': rootdir + 'gyro_experiment_tracking/motor_tracking/motor_tracking/zconvolve_dark_gyro2.png',
    'min_max_val': [0.40, 0.05],  # [0.0, .5],  # normally is [0.1, 0.4], trying something new to catch wonky particles
    # roi is either None (no roi), a numpy array (the roi), or 'click' (which tells hough_track to define it here)
    'roi': 'click',
    'diff': False,
    'mask': False,
    'centroid_clipminmax': [0., 1.],
    'subtract_mean': False,
    # if subtract_background > 0, use this as the window size as a fraction of the total frame width
    'normalize_background': 0.2,
    'refmask_thresh': 21,
    # Determine whether to click on reference points or get them from cf points
    'cf_reference_pts': 'click',  # 'cf_points', None
    'check_modii': 200,
    # If kill_bright_thres > 0, we take pixels brighter than 0.9 * max to zero
    'kill_bright_thres': -1,
    'cent_num_times': 2,
    # 'invert': True, ---> To invert, set min_max_val = [0.4, 0.05], ie [big, small]
    }
