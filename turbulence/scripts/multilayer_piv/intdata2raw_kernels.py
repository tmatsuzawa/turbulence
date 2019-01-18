import library.tools.rw_data as rw
import numpy as np
import matplotlib.pyplot as plt
import library.tools.process_data as process
import argparse
import os
import library.display.graph as graph


"""Convert interpolated time-averaged data into a raw file """


parser = argparse.ArgumentParser('Sort PIVLab outputs of multilayer PIV experiments')
parser.add_argument('-f', '--filepath', help='Location of interpolated time-averaged hdf5 file',
                    type=str,
                    default='/Volumes/bigraid/takumi/turbulence/3dprintedbox/multilayerPIV_Dp120mm_Do25p6mm/grid_data.h5'
                    )
parser.add_argument('-scale', '--scale', help='conversion (mm/px). default:0.20',
                    type=float,
                    default=0.20
                    )
parser.add_argument('-fps', '--fps', help='frame rate of recorded video. default: 2000.',
                    type=float,
                    default=2000.
                    )
parser.add_argument('-cutoff', '--cutoff', help='Energy cutoff in mm2/s2. default: 3x10^4',
                    type=float,
                    default=3.*10**6
                    )
args = parser.parse_args()


dir = os.path.split(args.filepath)[0]
filepath = args.filepath

data = rw.read_hdf5(filepath)
e_raw = np.asarray(data['energy'])
print e_raw.shape  # (z, x, y) is easy to work with dragonfly

# e_raw2 = np.swapaxes(e_raw, 0, 1)
# e = np.swapaxes(e_raw2, 1, 2)
e = e_raw
e = e * (args.fps*args.scale)**2


cutoffe = args.cutoff

print e.shape
print 'energy maximum, mean:', np.nanmax(e), np.nanmean(e)

# Make sure data does not contain np.nan or np.inf. if so replace with zero.
e_ma = np.ma.fix_invalid(e, fill_value=0)
e[:] = e_ma.data

# plt.imshow(e[..., 260], cmap='gray')
# plt.colorbar()
# plt.show()

e[e > cutoffe] = cutoffe
e_scaled = e / cutoffe * 255



# # check if energy descrepancy at small z and high z
# ee = np.nanmean(e_scaled, axis=1)
# colors = graph.get_color_list_gradient('red', 'blue', n=ee.shape[1])
# for i in range(ee.shape[1])[::21]:
#     fig, ax = graph.plot(range(ee.shape[0]), ee[:, i], color=colors[i], alpha=0.5)
#     fig, ax = graph.scatter(range(ee.shape[0]), ee[:, i], color=colors[i], alpha=0.5)
# graph.add_discrete_colorbar(ax, colors)
# plt.show()

r = np.linspace(1., 1.5, e.shape[1])
rr = np.tile(r, (e.shape[0], 1))

for i in range(e.shape[2]):
    e[..., i] *= rr



## Fix the resolution difference due to different z position
# for i in range(e.shape[2]):
#     graph.pdf(e_scaled[..., i], nbins=100)


deltafx = 0.04 # mm/px
fxmin, fx, fxmax = args.scale - deltafx, args.scale, args.scale + deltafx
kernel = np.linspace(fxmin, fxmax, e.shape[2]) ** 2
kernel = kernel[::-1]
kernel /= kernel[-1]

for i in range(e.shape[2]):
    e_scaled[..., i] /= kernel[i]

plt.plot(np.nanmean(e / cutoffe * 255, axis=(0, 1)), label='original')
plt.plot(np.nanmean(e_scaled, axis=(0, 1)), label='corrected')
plt.legend()
plt.show()


e_scaled = e_scaled.astype('uint8')

test = e_scaled[..., 50]
plt.imshow(test, cmap='gray')
plt.colorbar()
plt.show()




# e_scaled = np.delete(e_scaled, range(95, 112), 0)

output_path = dir + '/%s_%04d_%04d_%04d_cutoff_%08d_deltadx_%s_mod.raw'\
                    % (os.path.split(args.filepath)[1], e_scaled.shape[0], e_scaled.shape[1], e_scaled.shape[2],
                       args.cutoff, str(deltafx).replace('.', 'p'))
e_scaled.tofile(output_path)
print '... saved ', output_path

yc, xc = 610., 420.,

x, y = data['x'][..., 0], data['y'][..., 0]
A, sigma = 0.7, 200.
kernel2 = A * np.exp(- ((x-xc)**2+(y-yc)**2) / (2. * sigma)**2)

ymin, ymax = 40, 85
xmin, xmax = 60, 97
# print y[0, 40:80]
# print x[60:, 0]

kernel3 = np.ones_like(x)
kernel3[xmin:xmax, ymin:ymax] = 0.7

xc, yc = 750., 520.,
sigma = 150.
kernel4 = 1 - A * np.exp(- (0.1*(x-xc)**2+(y-yc)**2) / (2. * sigma**2))



e_mod = np.empty_like(e)
for i in range(e.shape[2]):
    print data['x'].shape
    fig, ax, cc = graph.color_plot(data['y'][..., i], data['x'][..., i], e[..., i] / kernel[i], vmin=0, vmax=1*10**4)
    ax.invert_yaxis()
    graph.add_colorbar(cc, option='scientific')
    graph.title(ax, 'z=%.3f' % data['z'][0, 0, i])
    graph.save(dir + '/mod_deltadx_%s/zm%03d' % (str(deltafx).replace('.', 'p'), i), ext='png')
    plt.close('all')

    if i < 100:
        fig, ax, cc = graph.color_plot(data['y'][..., i], data['x'][..., i], e[..., i] / kernel[i] * kernel4, vmin=0,
                                       vmax=10 * 10 ** 3)
        e_mod[..., i] = e[..., i] / kernel[i] * kernel4
    else:
        fig, ax, cc = graph.color_plot(data['y'][..., i], data['x'][..., i], e[..., i] / kernel[i], vmin=0,
                                       vmax=10 * 10 ** 3)
        e_mod[..., i] = e[..., i] / kernel[i]

    ax.scatter(yc, xc)
    ax.invert_yaxis()
    graph.add_colorbar(cc, option='scientific')
    graph.title(ax, 'z=%.3f' % data['z'][0, 0, i])
    graph.save(dir + '/mod_deltadx_%s_perspective_correction_bubble_corr_kernel/zm%03d' % (str(deltafx).replace('.', 'p'), i), ext='png')
    plt.close('all')

plt.plot(np.nanmean(e / cutoffe * 255, axis=(0, 1)), label='original')
plt.plot(np.nanmean(e_scaled, axis=(0, 1)), label='corrected')
plt.plot(np.nanmean(e_mod / cutoffe * 255, axis=(0, 1)), label='corrected2')
plt.legend()
plt.show()

e_mod[e_mod > cutoffe] = cutoffe
e_mod_scaled = e_mod / cutoffe * 255
e_mod_scaled = e_mod_scaled.astype('uint8')
output_path = dir + '/%s_%04d_%04d_%04d_cutoff_%08d_deltadx_%s_mod_corrected_using_kernel0_and_4.raw'\
                    % (os.path.split(args.filepath)[1], e_scaled.shape[0], e_scaled.shape[1], e_scaled.shape[2],
                       args.cutoff, str(deltafx).replace('.', 'p'))
e_mod_scaled.tofile(output_path)
print '... saved ', output_path

###49-24
e_scaled_small = e_scaled[..., 15:83]
i = 40
fig, ax, cc = graph.color_plot(data['y'][..., i], data['x'][..., i], e_scaled_small[..., i])
graph.add_colorbar(cc, ax=ax)
graph.show()

output_path = dir + '/%s_%04d_%04d_%04d_cutoff_%08d_deltadx_%s_mod_small.raw'\
                    % (os.path.split(args.filepath)[1], e_scaled_small.shape[0], e_scaled_small.shape[1], e_scaled_small.shape[2],
                       args.cutoff, str(deltafx).replace('.', 'p'))
e_scaled_small.tofile(output_path)
print '... saved ', output_path
###
e_mod_scaled_small = e_mod_scaled[..., 15:83]
output_path = dir + '/%s_%04d_%04d_%04d_cutoff_%08d_deltadx_%s_mod_corrected_using_kernel0_and_4_small.raw'\
                    % (os.path.split(args.filepath)[1], e_mod_scaled_small.shape[0], e_mod_scaled_small.shape[1], e_mod_scaled_small.shape[2],
                       args.cutoff, str(deltafx).replace('.', 'p'))
e_mod_scaled_small.tofile(output_path)
print e_mod_scaled_small.shape
print '... saved ', output_path