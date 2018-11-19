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


"""Sort PIVLab outputs of multilayer PIV experiments
    Make a hdf5 file for each slice"""


parser = argparse.ArgumentParser('Sort PIVLab outputs of multilayer PIV experiments')
parser.add_argument('-dir', '--dir', help='Directories where pivlab outputs lie',
                    type=str,
                    default='/Volumes/bigraid/takumi/turbulence/3dprintedbox/multilayerPIV_Dp57mm_Do12p8mm/2018_11_04/PIV_W8_step2_data/sample/PIVlab_Win64pix_W8px_Dt1_step2_PIV_fv_vp_left_macro105mm_fps2000_Dp56p57mm_D12p8mm_piston10p5mm_freq3Hz_v400mms_setting1_inj1p0s_trig5p0s_swvol2v_0')
parser.add_argument('-header', '--header', help='Header of PIVLab output files. Default: D.',
                    type=str, default='D')
parser.add_argument('-n', '--n', help='Number of slices used for multilayer piv experiment. Default: 17',
                    type=int, default=17)
parser.add_argument('-ntrig', '--ntrig', help='Number of times a camera was triggered at each illuminated plane. Default: 3',
                    type=int, default=3)
parser.add_argument('-m', '--m', help='Number of images per cycle. Should be n * ntrig / 0.85. Default 60.',
                    type=int, default=60)

args = parser.parse_args()
# preparation
nheader = len(args.header)# number of letters in a header

# make new directories
for i in range(args.n):
    resultdir = os.path.join(args.dir, 'slice%02d' % i)
    if not os.path.exists(resultdir):
        os.mkdir(resultdir)

# Glob all pivlab outputs
files = glob.glob(os.path.join(args.dir, args.header) + '*')
nfiles = len(files)
print nfiles

if nfiles !=0:
    # get a minimum value
    sequence = []
    for i, fyle in enumerate(files):
        filename = fs.get_filename_wo_ext(fyle)
        sequence.append(int(filename[nheader:]))
    start = min(sequence)
    if start != 1:
        print 'Are you resuming the sorter from somewhere? If so, continue. '
        start = input('What is the file no. of the first pivlab output on the Slice00? Probably 1 :')


    for i in tqdm(range(nfiles)):
        pdir, filename = os.path.split(files[i])
        filename_wo_ext = os.path.splitext(filename)[0]
        sliceno = (int(filename_wo_ext[nheader:]) % args.m - start) / args.ntrig
        # move the file
        resultdir = os.path.join(args.dir, 'slice%02d' % sliceno)
        os.rename(files[i], os.path.join(resultdir, filename))
    print 'Moved files to correct slice-numbered directories'
else:
    print '... Could not find pivlab outputs!'




### Generate hdf5 files
slicedirs = glob.glob(os.path.join(args.dir, 'slice') + '*')
for i, slicedir in enumerate(slicedirs):
    pivlab2hdf5.pivlab2hdf5(slicedir)