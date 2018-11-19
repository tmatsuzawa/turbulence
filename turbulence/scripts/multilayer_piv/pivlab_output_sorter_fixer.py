
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


"""Sort PIVLab outputs of multilayer PIV experiments
    Make a hdf5 file for each slice"""


parser = argparse.ArgumentParser('Sort PIVLab outputs of multilayer PIV experiments')
parser.add_argument('-dir', '--dir', help='Directories where pivlab outputs lie',
                    type=str,
                    default='/Volumes/bigraid/takumi/turbulence/3dprintedbox/multilayerPIV_Dp57mm_Do12p8mm/2018_11_04/PIV_W8_step2_data/PIVlab_Win64pix_W8px_Dt1_step2_PIV_fv_vp_left_macro105mm_fps2000_Dp56p57mm_D12p8mm_piston10p5mm_freq3Hz_v400mms_setting1_inj1p0s_trig5p0s_swvol2v_File')
parser.add_argument('-header', '--header', help='Header of PIVLab output files. Default: D.',
                    type=str, default='D')
parser.add_argument('-dirheader', '--dirheader', help='Header of sorted directories Default: slice',
                    type=str, default='slice')
parser.add_argument('-n', '--n', help='Number of slices',
                    type=int, default=17)
parser.add_argument('-m', '--m', help='Number of images per cycle. Should be n * ntrig / 0.85. Default 60.',
                    type=int, default=60)
parser.add_argument('-ntrig', '--ntrig', help='Number of times a camera was triggered at each illuminated plane. Default: 3',
                    type=int, default=3)

args = parser.parse_args()
# preparation
nheader = len(args.header)# number of letters in a header
npercycle = args.m

sorteddirs = glob.glob(os.path.join(args.dir, args.dirheader + '*'))
for sorteddir in sorteddirs:
    files = glob.glob(os.path.join(sorteddir, args.header + '*'))
    sliceno = os.path.split(sorteddir)[1]
    nfiles = len(files)
    print 'Fixing sorting for ' + sliceno
    for i in tqdm(range(nfiles)):
        filename = fs.get_filename_wo_ext(files[i])
        fileno = int(filename[nheader:])


        correct_sliceno = (fileno % npercycle - 1) / args.ntrig
        if sliceno != correct_sliceno: # if missorted,
            correct_filepath = os.path.join(args.dir, args.dirheader + '%02d/' % correct_sliceno + filename + '.txt')
            # print sliceno, correct_sliceno
            # print correct_filepath
            os.rename(files[i], correct_filepath)


### Generate hdf5 files
slicedirs = glob.glob(os.path.join(args.dir, 'slice') + '*')
for i, slicedir in enumerate(slicedirs):
    pivlab2hdf5.pivlab2hdf5(slicedir)