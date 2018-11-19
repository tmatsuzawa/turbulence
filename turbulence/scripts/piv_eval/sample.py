import library.display.graph as graph
import numpy as np
import matplotlib.pyplot as plt
import turbulence.jhtd.get as jhtd_get
import turbulence.jhtd.tools as jhtd_tools
import library.basics.formatarray as fa


# Load the single snapshot from the first timestep only
# Specify the data path
datadir = '/Volumes/labshared3-1/takumi/JHTD-sample/JHT_Database/Data/'
filename = 'isotropic1024fine_zl_1_yl_1024_xl_1024_coarse_t0_0_tl_1_y0_0_x0_0_z0_0_.h5'
# Specify a directory where you would like to save the analysis results
filepath_big = datadir + filename


# Plot Ux,Uy,Uz heatmap
# Read parameters (xl, yl, zl etc.) that you used to generate the cutout data
param = jhtd_get.get_parameters(filepath_big)
print param
dx = 2 * np.pi / 1024.
dy = dz = dx
dt = 0.002  # time separation between data points (10 DNS steps)
# dt_sim = 0.0002  #simulation time step
nu = 0.000185  # viscosity

T = [param['t0'] + dt * i for i in range(param['tl'])]
X = [param['x0'] + dx * i for i in range(param['xl'])]
Y = [param['y0'] + dy * i for i in range(param['yl'])]
Z = [param['z0'] + dz * i for i in range(param['zl'])]

ux0, uy0, uz0 = jhtd_tools.read_v_on_xy_plane_at_t0(filepath_big, z=0)

xx, yy = np.meshgrid(X, Y)

nrows_sub, ncolumns_sub = 16, 16
xx_coarse = fa.coarse_grain_2darr(xx, nrows_sub, ncolumns_sub)
yy_coarse = fa.coarse_grain_2darr(yy, nrows_sub, ncolumns_sub)
ux0_coarse = fa.coarse_grain_2darr(ux0, nrows_sub, ncolumns_sub)
uy0_coarse = fa.coarse_grain_2darr(uy0, nrows_sub, ncolumns_sub)
uz0_coarse = fa.coarse_grain_2darr(uz0, nrows_sub, ncolumns_sub)
#energy_coarse = (ux0_coarse ** 2 + uy0_coarse ** 2 + uz0_coarse ** 2) / 2
energy_coarse = np.sqrt((ux0_coarse ** 2 + uy0_coarse ** 2 + uz0_coarse ** 2))

#graph.color_plot(xx_coarse, yy_coarse, ux0_coarse, cmap='RdBu', vmin=-2, vmax=2, fignum=1)
print xx_coarse.shape
# # graph.color_plot(xx, yy, ux0, cmap='RdBu', vmin=-2, vmax=2, fignum=1)
fig = plt.figure(num=1, figsize=(18, 18))
ax = fig.add_subplot(111)
cc = ax.pcolormesh(xx_coarse, yy_coarse, uy0_coarse, cmap='RdBu', vmin=-2, vmax=2)
ax.quiver(xx_coarse, yy_coarse, ux0_coarse, uy0_coarse)
# ax.invert_yaxis()

ax.set_aspect('equal')
# set edge color to face color
cc.set_edgecolor('face')

graph.add_colorbar(cc, ax=ax)

plt.show()
