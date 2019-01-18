import library.display.graph as graph
import matplotlib.pyplot as plt
import numpy
import h5py

h5filepath = '/Volumes/bigraid/takumi/turbulence/3dprintedbox/multilayerPIV_Dp57mm_Do12p8mm/2018_11_16/PIV_W16_step2_data/time_avg_field_raw_portion_1.h5'

with h5py.File(h5filepath, 'r') as data:
    print dir(data)