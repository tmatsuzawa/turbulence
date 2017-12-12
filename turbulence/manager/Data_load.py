# -*- coding: utf-8 -*-
"""
Created on Tue May 26 09:57:15 2015
Generate all python structures to store the data associated to one particular date
Sdata : associated to each cine file. Contains the time stamp, spatial scale, origin in space, origin in time
Mdata : for each Dir of measurements, after PIV processing for instance
@author: stephane
"""

import matplotlib

matplotlib.use('Agg')
import sys
sys.path.append('/Users/stephane/Documents/git/takumi/turbulence/')
import turbulence.mdata.Sdata_manip as Sdata_manip
import turbulence.mdata.M_manip as M_manip

import argparse


parser = argparse.ArgumentParser(description="Create Sdata and Mdata objects associated to cine files and PIV measurements")
parser.add_argument('-d', dest='date', default='1987_03_19', type=str,
                    help='date to be processed. Python will look for it in all the folders specified in file_architecture.py')
parser.add_argument('-f', dest='folder', default=None, type=str,
                    help='base folder to be processed. Python will look for cine files inside this specified folder')
parser.add_argument('-s', dest='start', default=None, type=int, help='start processing index of the cinefile List')
parser.add_argument('-e', dest='end', default=None, type=int, help='end processing index of the cinefile List')
args = parser.parse_args()


def load(date):
    """
    Import data from a given date
    INPUT
    -----
    date : str
    OUTPUT
    -----
    Mlist : list of Mdata
        class Mdata in stephane.mdata.Mdata
        inherited classes : Mdata_PIVlab, Mdata_pyPIV
    """
    Mlist = M_manip.load_Mdata_serie(date)

    return Mlist


def process(date):
    setattr(args, 'date', date)
    main()


def process_list(date_list):
    for date in date_list:
        process(date)


def main():
    Sdata_manip.Sdata_gen_day(args.date)
    print '<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>'
    print 'Sdata is successfully created!'
    print '<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>'
    print 'Create Mdata using generated Sdata which is stored in hdf5'
    print '<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>'
    M_manip.Measure_gen_day(args.date)


if __name__ == '__main__':
    main()
