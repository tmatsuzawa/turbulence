import argparse
import movie_instance as mi
import library.basics.formatstring as fs
import h5py
import link_points as lp
import matplotlib.image as mpimg
import numpy as np
import os
import sys
import time
import tracking_helper_functions as thf
import experiment_movies
from settings import tracking_settings
import matplotlib.pyplot as plt
# from click_pts import ImagePoint
from basics.roipoly import RoiPoly
import glob
import cPickle as pkl

parser = argparse.ArgumentParser(description='Generate tiffs with tracking path from cines')
parser.add_argument('-cinefile', '--cinefile', help='cine file', type=str)
parser.add_argument('-fps', '--fps', help='frame rate', default=4000.0)

args = parser.parse_args()
cinefile = args.cinefile
frame_rate = args.fps

basedir, fn = os.path.split(cinefile)
output_dir = os.path.join(basedir, 'tracking_movie')


if __name__ == '__main__':
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    movie = mi.GyroMovie(fn, frame_rate=frame_rate)
    movie.save_frame_with_tracking_path()