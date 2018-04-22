""" Generate text images that can be loaded to imageJ from a pickle file for MPIV results
Run visualize_blob_4d_smallbox_t__.py to generate a data file
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
# import sys
# sys.path.append('/Users/stephane/Documents/git/takumi/noah2/basics/')
import os
import basics.dataio as dio
import basics.stringformat as sf
import glob
from scipy.interpolate import RegularGridInterpolator
import cPickle as pkl
import argparse
"""
Note: must be run with mayavi, so done in Canopy, for example.
"""

parser = argparse.ArgumentParser('Evolve a spring+gyro system on GPU forward then conjugate and reverse')
parser.add_argument('-overwrite', '--overwrite',
                    help='overwrite the existing interpolation',
                    action='store_true')
parser.add_argument('-method', '--method',
                    help='index for method from list (scatter, isosurfaces, or colorgrid) -> (0, 1, 2)',
                    type=int, default=0)
parser.add_argument('-dx', '--dx', help='Grid spacing in mm for image',
                    type=float, default=5.0)
parser.add_argument('-res', '--resolution', help='Resolution in dpi for image',
                    type=int, default=600)
args = parser.parse_args()


def sz_from_resolution(resolution=300):
    size_mm = 180.0
    size = (int(size_mm * resolution), int(size_mm * resolution))
    return size


# Set method as scatter, isosurfaces, or colorgrid
method_index = args.method
method = ['isosurfaces', 'scatter', 'colorgrid'][method_index]
overwrite = args.overwrite
# Note sqrt(L^2 + r^2) sin theta = r. We supply L here:
#                     r
#                    ____
#                    \  |
#                     \ |  L
#  halfangle_swept     \|
#                       .
#

# Define the distance of laser to center, in mm
dx = args.dx
ll = 712.8  # mm
halfangle_swept = 5.3 / 180. * np.pi
angle_swept = 2. * halfangle_swept
# Assume that slices span an angle (2 * Dtheta)
nslices = 17
dtheta = angle_swept / float(nslices - 1)
# Define width of the FOV
pix2mm = 1.0

seriesdir = '/Volumes/labshared3-1/takumi/2018_02_20_stacks1_noisy/AnalysisResults/Time_averaged_Plots_0/'
outputdir = seriesdir + 'imageJ_textimages/slices_interp3d_dx' + sf.float2pstr(dx) + '/'

# Make output directory if it does not exist
if not os.path.isdir(outputdir):
    os.mkdir(outputdir)

# Name of the pickle file to load
interpfn = seriesdir + 'slices_interp3d_dx' + sf.float2pstr(dx) + '.pkl'

# Check if the data file (pkl) exists
if os.path.exists(interpfn):
    fn = open(interpfn, 'rb')
    egrid = pkl.load(fn)
    fn.close()
    print egrid.shape
    print egrid[:][0][:].shape
    X, Y, Z = egrid.shape
    print X, Y, Z
    print egrid[X-1][Y-1][Z-1]

    for i in range(egrid.shape[0]):
        print i
        output_fn = outputdir + 'time_averaged_energy_z%03d' % i + '.txt'
        #output_file = open(output_fn, 'w')
        np.savetxt(output_fn, np.transpose(egrid[i][:][:]))

    #print egrid[:][0][:]




else:
    print (interpfn + ' does NOT exist! You need a pkl file first.')


print ('Done')