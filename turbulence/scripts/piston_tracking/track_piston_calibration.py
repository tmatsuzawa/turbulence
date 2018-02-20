import numpy as np
import glob

"""Process the tracking of the piston to calibrate its motion"""

cinepath = '/Volumes/labshared3/noah/turbulence/midsize_box_motor_calibration/'
cines = glob.glob(cinepath + '*.cine')

for cine in cines:
