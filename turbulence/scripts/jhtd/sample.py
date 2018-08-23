# Specify a data path
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
import turbulence.jhtd.get as jhtd_get
import turbulence.jhtd.tools as jhtd_tools




##Plot Ux,Uy,Uz heatmap
#Read parameters (xl, yl, zl etc.) that you used to generate a cutout data
param = jhtd_get.get_parameters(filepath)
print param
dx = 2*np.pi/1024.
dy=dz=dx
dt = 0.002 # time separation between data points (10 DNS steps)
#dt_sim = 0.0002  #simulation timestep
nu = 0.000185 #viscosity

T = [param['t0'] + dt*i for i in range(param['tl'])]
X = [param['x0'] + dx*i for i in range(param['xl'])]
Y = [param['y0'] + dy*i for i in range(param['yl'])]
Z = [param['z0'] + dz*i for i in range(param['zl'])]

ux0, uy0, uz0 = jhtd_tools.read_v_on_xy_plane_at_t0(filepath, z=0)

plt.figure(figsize=(32,24))
plt.subplot(1, 1, 1)
plt.pcolor(X, Y, ux0, cmap='RdBu',label='z=0')
cbar=plt.colorbar()
cbar.ax.set_ylabel('$U_x$', fontsize=75)
cbar.ax.tick_params(labelsize=50)
plt.xlabel('X',fontsize=75)
plt.ylabel('Y',fontsize=75)
plt.xticks(size=50)
plt.yticks(size=50)
# filename = 'Ux_colormap_z=0'
# filepath = savedir + filename
# graph.save(filepath, ext='png')


plt.figure(figsize=(32,24))
plt.subplot(1, 1, 1)
plt.pcolor(X, Y, uy0, cmap='RdBu',label='z=0')
cbar=plt.colorbar()
cbar.ax.set_ylabel('$U_y$', fontsize=75)
cbar.ax.tick_params(labelsize=50)
plt.xlabel('X',fontsize=75)
plt.ylabel('Y',fontsize=75)
plt.xticks(size=50)
plt.yticks(size=50)
# filename = 'Uy_colormap_z=0'
# filepath = savedir + filename
# graph.save(filepath, ext='png')


plt.figure(figsize=(32,24))
plt.subplot(1, 1, 1)
plt.pcolor(X, Y, uz0, cmap='RdBu',label='z=0')
cbar=plt.colorbar()
cbar.ax.set_ylabel('$U_z$', fontsize=75)
cbar.ax.tick_params(labelsize=50)
plt.xlabel('X',fontsize=75)
plt.ylabel('Y',fontsize=75)
plt.xticks(size=50)
plt.yticks(size=50)
# filename = 'Uz_colormap_z=0'
# filepath = savedir + filename
# graph.save(filepath, ext='png')

plt.show()
