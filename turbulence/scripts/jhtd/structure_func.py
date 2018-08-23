# Specify the data path
datadir = '/Volumes/labshared3-1/takumi/JHTD-sample/JHT_Database/Data/'
filename = 'isotropic1024fine_zl_1_yl_1024_xl_1024_coarse_t0_0_tl_1_y0_0_x0_0_z0_0_.h5'


# Specify a directory where you would like to save the analysis results
filepath = datadir + filename
savedir= datadir + 'AnalysisResults/'

# Import modules
# import sys
# sys.path.append('/Users/stephane/Documents/git/takumi/turbulence')
import os
import glob
import numpy as np

import matplotlib.pyplot as plt
import h5py

import turbulence.jhtd.get as jhtd_get
import turbulence.jhtd.jhtddata as jhtddata
import turbulence.jhtd.cutout as jhtd_cutout
import turbulence.jhtd.tools as jhtd_tools
import library.display.graph as graph
import library.tools.rw_data as rw

def compute_struc_func(ux0, white_noise, xx, yy, n_bins, p=2, option='original'):
    rr, DLL = np.zeros(xx.shape[0] * xx.shape[1] * n_bins), np.zeros(xx.shape[0] * xx.shape[1] * n_bins)
    for i in range(n_bins):  # in x direction
        if i % 50 == 0:
            print '%d / %d' % (i + 1, n_bins)
        xx_rolled = np.roll(xx, i, axis=0)
        yy_rolled = np.roll(yy, i, axis=0)

        rr_raw = np.sqrt((xx - xx_rolled) ** 2 + (yy - yy_rolled) ** 2)
        rr[i * xx.shape[0] * xx.shape[1]:(i + 1) * xx.shape[0] * xx.shape[1]] = rr_raw.flatten()

        if option == 'original':
            ux_rolled = np.roll(ux0, i, axis=0)
            DLL_raw = (ux0 - ux_rolled) ** p
            DLL[i * xx.shape[0] * xx.shape[1]:(i + 1) * xx.shape[0] * xx.shape[1]] = DLL_raw.flatten()
        elif option == 'crossterm':
            ux_rolled = np.roll(ux0, i, axis=0)
            noise_rolled = np.roll(white_noise, i, axis=0)
            DLL_raw = 2 * (ux0 - ux_rolled) * (white_noise - noise_rolled)  # cross-term
            DLL[i * xx.shape[0] * xx.shape[1]:(i + 1) * xx.shape[0] * xx.shape[1]] = DLL_raw.flatten()
        elif option == 'noise':
            noise_rolled = np.roll(white_noise, i, axis=0)
            DLL_raw = (white_noise - noise_rolled) ** p
            DLL[i * xx.shape[0] * xx.shape[1]:(i + 1) * xx.shape[0] * xx.shape[1]] = DLL_raw.flatten()
    if option == 'original':
        print 'binning 1/2...'
        bin_centers, _, _ = binned_statistic(rr, rr, statistic='mean', bins=n_bins)
    else:
        bin_centers = None
    print 'binning 2/2...'
    bin_averages, _, _ = binned_statistic(rr, DLL, statistic='mean', bins=n_bins)
    return bin_centers, bin_averages


#Read parameters (xl, yl, zl etc.) that you used to generate the cutout data
param = jhtd_get.get_parameters(filepath)
print param
dx = 2*np.pi/1024.
dy=dz=dx
dt = 0.002 # time separation between data points (10 DNS steps)
#dt_sim = 0.0002  #simulation timestep
nu = 0.000185 #viscosity
epsilon = 0.103
eta = (nu**3 / epsilon) ** 0.25

T = [param['t0'] + dt*i for i in range(param['tl'])]
X = [param['x0'] + dx*i for i in range(param['xl'])]
Y = [param['y0'] + dy*i for i in range(param['yl'])]
Z = [param['z0'] + dz*i for i in range(param['zl'])]

ux0, uy0, uz0 = jhtd_tools.read_v_on_xy_plane_at_t0(filepath, z=0)
yy, xx = np.meshgrid(X,Y)



# Compute signal-to-noise correlation etc.

# add noise using gaussian
#sigmas = [ux0*0.1, ux0*0.2, ux0*0.4, ux0*0.5]
mu = 0

# add white noise
mag = 0.844

white_noise = mag * (np.random.random(ux0.shape) - 0.5)
ux0_noise = (ux0 * (1 + white_noise))
snr = np.var(ux0) / np.var(ux0*white_noise)
print 'SNR: %.2f' % snr

# structure function calculation
from scipy.stats import binned_statistic
# order of structure function
p = 2

#################
n_bins = xx.shape[0]/1 # Dividing this by 2 is for a shorter computation time.
#################
options = ['original', 'crossterm', 'noise']
labels = ['original', 'signal-noise corr.', 'noise autocorr.']
title = 'SNR: 10.00'
data = {}
for i, option in enumerate(options):
    print labels[i]
    if i == 0:
        rr, DLL = compute_struc_func(ux0, ux0_noise, xx, yy, n_bins, option=option)
    else:
        _, DLL = compute_struc_func(ux0, ux0_noise, xx, yy, n_bins, option=option)
    rr_scaled = rr / eta
    Dxx_scaled = DLL / ((epsilon*rr)**(2./3))
    fig, ax = graph.plot(rr_scaled, Dxx_scaled, label=labels[i], alpha=0.9)
    fig, ax = graph.scatter(rr_scaled, Dxx_scaled, alpha=0.9)

    data['rr_scaled_' + option] = rr_scaled
    data['Dxx_scaled_' + option] = Dxx_scaled
    data['rr_' + option] = rr
    data['Dxx_' + option] = DLL

graph.tosemilogx(ax)
ax.legend(loc=2)
graph.labelaxes(ax, '$r/\eta$', r'$D_{xx}/(\epsilon r)^{2/3}$')
#graph.setaxes(ax, 1, 5000, -0.2, 4)
graph.axhband(ax, 2*0.85, 2*1.15, 1, 5000)
graph.save(savedir + 'jhtd_struc_func_scaled_white_noise_budget_test')
# graph.show()

datafile = 'jhtd_struc_func_scaled_white_noise_budget_test_data'
rw.write_hdf5_dict(savedir + datafile, data)