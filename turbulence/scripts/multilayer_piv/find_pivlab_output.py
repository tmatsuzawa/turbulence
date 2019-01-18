import scipy
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
import library.basics.formatstring as fs
import turbulence.manager.pivlab2hdf5 as pivlab2hdf5
import os
import math
import sys
from tqdm import tqdm


"""Finds a raw pivlab output in sorted directories"""


parser = argparse.ArgumentParser('Sort PIVLab outputs of multilayer PIV experiments')
parser.add_argument('-dir', '--dir', help='Directories where pivlab outputs lie',
                    type=str,
                    default='/Volumes/bigraid/takumi/turbulence/3dprintedbox/multilayerPIV_Dp57mm_Do12p8mm/2018_11_04/PIV_W8_step2_data/PIVlab_Win64pix_W8px_Dt1_step2_PIV_fv_vp_left_macro105mm_fps2000_Dp56p57mm_D12p8mm_piston10p5mm_freq3Hz_v400mms_setting1_inj1p0s_trig5p0s_swvol2v_right30mm_File')
parser.add_argument('-header', '--header', help='Header of PIVLab output files. Default: D.',
                    type=str, default='D')
parser.add_argument('-dirheader', '--dirheader', help='Header of sorted directories Default: slice',
                    type=str, default='slice')
parser.add_argument('-nslice', '--nslice', help='Number of slices',
                    type=int, default=17)
parser.add_argument('-n', '--n', help='File number',
                    type=int, default=17)
parser.add_argument('-ext', '--ext', help='File extension',
                    type=str, default='txt')
args = parser.parse_args()

filename = args.header + '%06d' % args.n + '.' + args.ext

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)



sorteddirs = glob.glob(os.path.join(args.dir, args.dirheader + '*'))
for sorteddir in sorteddirs:
    result = find(filename, sorteddir)
    print result



