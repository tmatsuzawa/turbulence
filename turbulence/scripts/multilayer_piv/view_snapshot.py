import h5py
import numpy as np
import matplotlib.pyplot as plt
import library.display.graph as graph

griddata_path = '/Volumes/bigraid/takumi/turbulence/3dprintedbox/multilayerPIV_Dp57mm_Do12p8mm/2018_11_04/PIV_W8_step2_data/sample/3dinterpolated_data_200x200x200_center_linear.h5'
resultdir = '/Volumes/bigraid/takumi/turbulence/3dprintedbox/multilayerPIV_Dp57mm_Do12p8mm/2018_11_04/PIV_W8_step2_data/sample/results_linear/'
with h5py.File(griddata_path,'r') as data:
    uxdata = np.asarray(data['ux'])
    uydata = np.asarray(data['uy'])
    energy = (uxdata**2 + uydata**2) / 2. * 10**-6 # m^2/s^2
    xx, yy, zz = np.asarray(data['x']), np.asarray(data['y']), np.asarray(data['z'])
    for i in range(uxdata.shape[2]):
        fig, ax, cc = graph.color_plot(xx[..., 0], yy[..., 0], energy[..., i, 0], cmap='plasma', vmin=0, vmax=0.06)
        graph.add_colorbar(cc, ax=ax)
        fig.tight_layout()
        graph.save(resultdir + 'im%04d' % i, ext='png')
        # graph.save(resultdir + 'pdf/im%04d' % i, ext='pdf')
        plt.close('all')