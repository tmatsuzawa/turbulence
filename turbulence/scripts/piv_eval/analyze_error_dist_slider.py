"""
experimenting with Bokeh for interactive plotting. Did not quite work since there are too many data points to keep track of
"""


import numpy as np
import h5py
import glob
import library.basics.formatstring as fs
import library.basics.formatarray as fa
import library.basics.std_func as std_func
import library.display.graph as graph
import numpy as np
import matplotlib as mpl
from scipy.optimize import curve_fit
from bokeh.layouts import row, widgetbox
from bokeh.models import CustomJS, Slider
from bokeh.plotting import figure, output_file, show, ColumnDataSource


def search_tuple(tups, elem):
    """
    Searches an element in tuple, and returns a list of found tuple elements

    Example:
    tuples = [(1, "hey"), (2, "hey"), (3, "no")]
    print(search_tuple(tuples, "hey"))
    >> [(1, "hey"), (2, "hey")]
    print(search_tuple(tuples, 3))
    >> [(3, "no")]

    Parameters
    ----------
    tups
    elem

    Returns
    -------

    """
    return filter(lambda tup: elem in tup, tups)

# Data architecture
dir = '/Volumes/bigraid/takumi/turbulence/JHTD/synthetic_data/hdf5data/tstep1_npt50000_lt20p0_pbcTrue_varyingDt/post_processed/err_data'
errdata_list = glob.glob(dir + '/*.h5')
errdata_list = fa.natural_sort(errdata_list)

# jhtd parameters
# dx = 2 * np.pi / 1024. # unit length / px
dy = dz = dx = 1  # in px
dt_sim = 0.0002  # DNS simulation time step

dt_spacing = 10
# dt = dt_sim * param['tstep'] * dt_spacing # time separation between data points in JHTD time unit
nu = 0.000185  # viscosity (unit length ^2 / unit time)
fx = (2 * np.pi) / 1024  # unit length / px
tau = 0.19 # time scale of out-of-plane motion

# Plotting settings 1
params = {'figure.figsize': (20, 20),
          'font.size': 20,
          'legend.fontsize': 20,
          'axes.labelsize': 20}
lw = 4 # line width of plots
graph.update_figure_params(params)
cmap = mpl.cm.get_cmap('magma')
normalize = mpl.colors.Normalize(vmin=0, vmax=0.20)


for i, errdata_path in enumerate(errdata_list):
    if i>0:
        break
    print errdata_path
    errdata = h5py.File(errdata_path, mode='r')

    keys_all = [u'ux_center_temp', u'ux_err_center_temp', u'ux_err_mean_temp', u'ux_mean',
                u'uy_center_temp', u'uy_err_center_temp', u'uy_err_mean_temp', u'uy_mean']

    keys = [u'ux_mean',  u'ux_err_mean_temp', u'uy_mean', u'uy_err_mean_temp']
    titles = [r'$W=8$px', '$W=16$px', '$W=32$px', '$W=64$px']
    suptitles = [r'$\Delta U_x$', r'$\Delta U_{x,center}$', r'$\Delta U_y$', r'$\Delta U_{y,center}$']
    iws = [8, 16, 32, 64]
    subplots = [221, 222, 223, 224]
    subplot_tpl = zip(iws, subplots)

    deltat = fs.get_float_from_str(errdata_path, 'Dt_',
                                   '_')  # number of DNS steps between image A and image B = deltat * 10 for isotropic1024coarse
    iw = fs.get_float_from_str(errdata_path, 'W', 'pix')  # interrogation window size in px
    lt = fs.get_float_from_str(errdata_path, 'lt', '_')  # laser thickness in px


    # Plotting settings 2
    subplot = search_tuple(subplot_tpl, iw)[0][1]  # 3 digit subplot index obtained from subplot_tpl
    # label = '$\Delta t$=%d DNS steps = %.3f (a.u)' % (deltat * dt_spacing, deltat * dt_spacing * dt_sim)
    label = '$\Delta t = %.3f $ (a.u)' % (deltat * dt_spacing * dt_sim)

    vmax = 1024 / 2 * dt_sim * (deltat * dt_spacing)  # px/frame
    # vmax = iw * 8 / 2
    vmin = - vmax


    datadict = {}
    for key in keys:
        datadict[key] = np.asarray(errdata[key]).flatten()

    for key in datadict:
        print key, datadict[key].shape

    # Bokeh
    source = ColumnDataSource(datadict)
    plot = figure(plot_width=1000, plot_height=1000)
    plot.scatter('ux_mean', 'ux_err_mean_temp', source=source, line_width=3, line_alpha=0.3)

    layout = row(
        plot,
        # widgetbox(amp_slider, freq_slider, phase_slider, offset_slider),
    )

    output_file("W%d_deltat%d_slider.html" % (iw, deltat), title="slider.py example")

    show(layout)
#
# x = np.linspace(0, 10, 500)
# y = np.sin(x)
# y2 = np.cos(x)
# datadict = {}
# keys = ['x', 'y', 'y2']
# data = [x, y, y2]
# for key, datum in zip(keys, data):
#     datadict[key] = datum
#
# source = ColumnDataSource(datadict)
#
# plot = figure(y_range=(-10, 10), plot_width=600, plot_height=600)
# plot2 = figure(y_range=(-10, 10), plot_width=600, plot_height=600)
#
#
# plot.scatter('x', 'y', source=source, line_width=3, line_alpha=0.6)
# plot2.scatter('x', 'y2', source=source, line_width=3, line_alpha=0.6)
#
#
# callback = CustomJS(args=dict(source=source), code="""
#     var data = source.data;
#     var A = amp.value;
#     var k = freq.value;
#     var phi = phase.value;
#     var B = offset.value;
#     var x = data['x']
#     var y = data['y']
#     var y2 = data['y2']
#     for (var i = 0; i < x.length; i++) {
#         y[i] = B + A*Math.sin(k*x[i]+phi);
#         y2[i] = B + A*Math.cos(k*x[i]+phi);
#     }
#     source.change.emit();
# """)
#
#
#
#
# amp_slider = Slider(start=0.1, end=10, value=1, step=.1,
#                     title="Amplitude", callback=callback)
# callback.args["amp"] = amp_slider
#
# freq_slider = Slider(start=0.1, end=10, value=1, step=.1,
#                      title="Frequency", callback=callback)
# callback.args["freq"] = freq_slider
#
# phase_slider = Slider(start=0, end=6.4, value=0, step=.1,
#                       title="Phase", callback=callback)
# callback.args["phase"] = phase_slider
#
# offset_slider = Slider(start=-5, end=5, value=0, step=.1,
#                        title="Offset", callback=callback)
# callback.args["offset"] = offset_slider
#
# layout = row(
#     plot,
#     plot2,
#     widgetbox(amp_slider, freq_slider, phase_slider, offset_slider),
# )
#
# output_file("slider.html", title="slider.py example")
#
# show(layout)