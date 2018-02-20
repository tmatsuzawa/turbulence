# input_dir = '/Volumes/GetIt/saved_stuff/2017_03_07_after_fix/obstacle/5p02_amps.cine'
# tracking_settings = {
#     'input_dir': input_dir,
#     'output_dir': '/Volumes/GetIt/saved_stuff/2017_03_07_after_fix/obstacle/',
#     'first_frame': 0,
#     'last_frame': -1,
#     'pix': 6,
#     'min_max_radius': [17, 24],
#     #'tracked_image': -1,
#     'cf': [0],
#     'min_max_val': [0.15, 0.4]
#     }

# input_dir = '/Volumes/labshared2/noah/201708_pwmtlc_tests/20170806_pwmcalibration/54gyro/9v_4p9vled_2p2kref/'
# input_dir = '/Volumes/labshared3/noah/201709_experiments/minis/'

# input_dir = '/Volumes/research2TB/gyroab_experiment/201709_experiments/20170907_sweepfreq_vs_coilv_gap/'
# input_dir = '/Volumes/labshared3/noah/201709_experiments/20170922_sweepfreq_175hz/'
# input_dir = '/Volumes/labshared3/noah/201709_experiments/20170922_calib_vab_displacement_175hz/'

# input_dir = '/Volumes/labshared3/noah/201709_experiments/20171007_switchoff_2p0hz_175hz/'
# input_dir = '/Volumes/research2TB/gyroab_experiment/201709_experiments/20171007_switchoff_2p0hz_175hz/'
# input_dir = '/Volumes/research2TB/gyroab_experiment/201709_experiments/20171007_pulsepacket_vs_coilv/'
# input_dir = '/Volumes/research2TB/gyroab_experiment/201709_experiments/test/'
# input_dir = '/Volumes/labshared3/noah/201709_experiments/20171005_calib_vab_omg_175hz/'
# input_dir = '/Volumes/research2TB/gyroab_experiment/201709_experiments/20171005b_calib_vab_omg_175hz/'
# input_dir = '/Volumes/research2TB/gyroab_experiment/201709_experiments/20171005b_nogood/'
# input_dir = '/Volumes/research2TB/gyroab_experiment/201709_experiments/20171005_calib_vab_omg_175hz/'
# input_dir = '/Volumes/research2TB/gyroab_experiment/201709_experiments/20170922_switchoff_2p0hz_175hz/'
# input_dir = '/Volumes/labshared3/noah/201709_experiments/20170907_sweepfreq_vs_coilv_gap/'
# input_dir = '/Volumes/labshared3/noah/201709_experiments/20170907_pulsepacket_vs_coilv_longerexp/'

# input_dir = '/Volumes/labshared3/noah/201709_pwmtlc_tests/20171020_pwmtlc_equilize_54gyro/pwm1/'
# input_dir = '/Volumes/labshared3/noah/201709_pwmtlc_tests/20171019_direct_equilize_2gyro/'
# input_dir = '/Volumes/labshared3/noah/201709_experiments/20171011_omg_vs_omegafreq_1gyro/omega_precession/selection/'
# input_dir = '/Volumes/labshared3/noah/201709_experiments/20171011_omg_vs_omegafreq_1gyro/omega_spinning/selection/'
# input_dir = '/Volumes/research2TB/gyroab_experiment/201709_experiments/20171010_aoverl_measurements/'
# input_dir = '/Volumes/research2TB/gyroab_experiment/201709_experiments/20170922_calib_vab_displacement_175hz/'
# input_dir = '/Volumes/labshared3/noah/201709_experiments/20171013_spectrum_measurement/'
# input_dir = '/Volumes/labshared3/noah/201709_experiments/20171019_spectrum_measurement_2gyro_405hz/'
# input_dir = '/Volumes/labshared3/noah/201709_experiments/20171020_calib_vab_displacement_175hz/'

# input_dir = '/Volumes/labshared3/noah/201709_experiments/20171013_omg_for_spectrum_measurement_2gyro_300hz/'
# input_dir = '/Volumes/labshared3/noah/201709_experiments/20171013_spectrum_measurement_3gyro_175hz/'
# input_dir = '/Volumes/labshared3/noah/201709_experiments/20171006_spectrum_measurement_2gyro/'
# input_dir = '/Volumes/labshared3/noah/201709_experiments/20171010_spectrum_measurement_2gyro/'

# input_dir = '/Volumes/labshared3/noah/201709_experiments/20171010_aoverl_measurements/'
# input_dir = '/Volumes/labshared3/noah/201709_experiments/20170904_sweepfreq_vs_coilv/'

# input_dir = '/Volumes/labshared3/noah/201709_experiments/minis/'
# input_dir = '/Volumes/labshared3/noah/201709_experiments/20170901_1p9hz_locz/'
# input_dir = '/Volumes/research2TB/minis/'
# input_dir = '/Volumes/labshared2/noah/201708_springcalibration/'


# TAI TRANSITION
option = 0
input_dir = '/Volumes/labshared3/noah/turbulence/midsize_box_motor_calibration/'
# input_dir = '/Volumes/labshared3/noah/201801_TAI_experiments/20180204_tai_54gyro/'
rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/'


# Notes
# roi is for selecting initial identification of particles only
#   it is not a mask applied to each frame (reference_mask)
# reference_mask is a mask applied to each frame before identification
#   this is useful for cropping out stationary bright spots that are sometimes near the gyro center


# single gyro settings
# tracking_settings = {
#     'input_dir': input_dir,
#     'output_dir': input_dir,
#     'first_frame': 0,
#     'last_frame': -1,
#     'pix': 10,
#     'min_max_radius': [17, 22],
#     # 'tracked_image': -1,
#     'cf': [0],  # frames to identify particles with Hough Transform --> list of ints or ['all']
#     'min_max_val': [0.4, 1.0]
#     }
#
# # many-gyro settings
# tracking_settings = {
#     'input_dir': input_dir,
#     'output_dir': input_dir,
#     'first_frame': 0,
#     'last_frame': -1,
#     'pix': 10,
#     'min_max_radius': [17, 22],
#     # 'tracked_image': -1,
#     'cf': [0],  # frames to identify particles with Hough Transform --> list of ints or ['all']
#     'min_max_val': [0.1, 0.4]
#     }

if option == 0:
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
