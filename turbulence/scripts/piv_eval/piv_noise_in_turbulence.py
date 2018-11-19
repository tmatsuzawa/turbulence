import h5py
import glob
import library.basics.formatstring as fs
import library.basics.formatarray as fa
import library.basics.std_func as std_func
import library.display.graph as graph
import numpy as np
import matplotlib as mpl
from scipy.optimize import curve_fit

# Data architecture
dir = '/Volumes/bigraid/takumi/turbulence/JHTD/synthetic_data/hdf5data/tstep1_npt50000_lt20p0_pbcTrue_varyingDt/post_processed/pdf_data_local'
histdata_list = glob.glob(dir + '/*.h5')
histdata_list = fa.natural_sort(histdata_list)

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


ax1s, ax2s, ax3s, ax4s = [], [], [], []
# fitting params
## 1,2,3,4, corresponds to ux_avg, ux_center, uy_avg, uy_center
deltat_all = [[], [], [], []]
gamma_all, chi_all = [[], [], [], []], [[], [], [], []]
iw_all = [[], [], [], []]
fit_err_all = [[], [], [], []]
r2s = [[], [], [], []]
for i, histdata_path in enumerate(histdata_list):
    # if i>0:
    #     break
    print histdata_path
    histdata = h5py.File(histdata_path, mode='r')

    keys_all = [u'bins_ux_diff_avg', u'bins_ux_diff_c', u'bins_ux_jhtd_avg', u'bins_ux_jhtd_c', u'bins_ux_pivlab',
                u'bins_uy_diff_avg', u'bins_uy_diff_c', u'bins_uy_jhtd_avg', u'bins_uy_jhtd_c', u'bins_uy_pivlab',
                u'hist_ux_diff_avg', u'hist_ux_diff_c', u'hist_ux_jhtd_avg', u'hist_ux_jhtd_c', u'hist_ux_pivlab',
                u'hist_uy_diff_avg', u'hist_uy_diff_c', u'hist_uy_jhtd_avg', u'hist_uy_jhtd_c', u'hist_uy_pivlab']
    # make all datasets numpy arrays


    keys = [u'bins_ux_diff_avg', u'bins_ux_diff_c', u'bins_uy_diff_avg', u'bins_uy_diff_c',
            u'hist_ux_diff_avg', u'hist_ux_diff_c', u'hist_uy_diff_avg', u'hist_uy_diff_c']
    titles = [r'$W=8$px', '$W=16$px', '$W=32$px', '$W=64$px']
    suptitles = [r'$\Delta U_x$', r'$\Delta U_{x,center}$', r'$\Delta U_y$', r'$\Delta U_{y,center}$']
    iws = [8, 16, 32, 64]
    subplots = [221, 222, 223, 224]
    subplot_tpl = zip(iws, subplots)

    deltat = fs.get_float_from_str(histdata_path, 'Dt_',
                                   '_')  # number of DNS steps between image A and image B = deltat * 10 for isotropic1024coarse
    iw = fs.get_float_from_str(histdata_path, 'W', 'pix')  # interrogation window size in px
    lt = fs.get_float_from_str(histdata_path, 'lt', '_')  # laser thickness in px


    # Plotting settings 2
    subplot = search_tuple(subplot_tpl, iw)[0][1]  # 3 digit subplot index obtained from subplot_tpl
    # label = '$\Delta t$=%d DNS steps = %.3f (a.u)' % (deltat * dt_spacing, deltat * dt_spacing * dt_sim)
    label = '$\Delta t = %.3f $ (a.u)' % (deltat * dt_spacing * dt_sim)

    vmax = 1024 / 2 * dt_sim * (deltat * dt_spacing)  # px/frame
    # vmax = iw * 8 / 2
    vmin = - vmax


    for j, key in enumerate(keys):
        if j > len(keys) / 2 - 1:
            break

        # # Data was stored in a strange manner, here is a fix to compare data.
        # if len(histdata[key].shape) != 1:
        #     bindatum = np.transpose(histdata[key])[:, 0]
        #     histdatum = np.transpose(histdata[keys[j + len(keys) / 2]])[:, 0]
        # else:
        #     bindatum = np.asarray(histdata[key])
        #     histdatum = np.asarray(histdata[keys[j + len(keys) / 2]])

        # Data was stored in a strange manner, here is a fix to compare data.
        print histdata[key].shape
        sys.exit()
        if len(histdata[key].shape) != 1:
            bindatum = np.transpose(histdata[key])[:, 0]
            histdatum = np.transpose(histdata[keys[j + len(keys) / 2]])[:, 0]
        else:
            bindatum = np.asarray(histdata[key])
            histdatum = np.asarray(histdata[keys[j + len(keys) / 2]])

        if key in [u'bins_ux_diff_avg']:
            print histdata[key].shape, histdata[keys[j+len(keys)/2]].shape
            fig1, ax1 = graph.plot(bindatum, histdatum,
                                   alpha=0.7, label=label, fignum=1, subplot=subplot,
                                   color=cmap(normalize((deltat * dt_spacing * dt_sim))), lw=lw)
            # graph.plot_fit_curve(np.array(histdata[key]) / vmax, np.array(histdata[keys[j+len(keys)/2]]) * vmax, std_func.lorentzian_norm, fignum=1, subplot=subplot)
            ind = 0
        elif key in [u'bins_ux_diff_c']:
            fig2, ax2 = graph.plot(bindatum, histdatum,
                                   alpha=0.7, label=label, fignum=2, subplot=subplot,
                                   color=cmap(normalize((deltat * dt_spacing * dt_sim))), lw=lw)
            ind = 1
        elif key in [u'bins_uy_diff_avg']:
            fig3, ax3 = graph.plot(bindatum, histdatum,
                                   alpha=0.7, label=label, fignum=3, subplot=subplot,
                                   color=cmap(normalize((deltat * dt_spacing * dt_sim))), lw=lw)
            ind = 2
        elif key in [u'bins_uy_diff_c']:
            fig4, ax4 = graph.plot(bindatum, histdatum,
                                   alpha=0.7, label=label, fignum=4, subplot=subplot,
                                   color=cmap(normalize((deltat * dt_spacing * dt_sim))), lw=lw)
            ind = 3

        if not ax1 in ax1s:
            ax1s.append(ax1)
        elif not ax2 in ax2s:
            ax2s.append(ax2)
        elif not ax3 in ax3s:
            ax3s.append(ax3)
        elif not ax4 in ax4s:
            ax4s.append(ax4)
        # curve fittings
        cond = np.abs(bindatum) < 10
        popt, pcov = curve_fit(std_func.lorentzian_norm, bindatum[cond], histdatum[cond])
        # Compute R2 values for goodness of fit
        y, y_fit = histdatum, std_func.lorentzian_norm(bindatum, popt[0], popt[1])
        ss_res = np.sum((y - y_fit) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        gamma_all[ind].append(popt[1])
        chi_all[ind].append(popt[0])
        deltat_all[ind].append(deltat)
        iw_all[ind].append('W%d' % int(iw))
        fit_err_all[ind].append(np.sqrt(np.diag(pcov)))
        r2s[ind].append(r2)
        print popt[1], popt[0]

    histdata.close()

for ax_pack in [ax1s, ax2s, ax3s, ax4s]:
    for ax, title in zip(ax_pack, titles):
        ax.set_xlim(-20, 20)
        # ax.set_ylim(0, 15)
        # ax.set_yscale("log")

        ax.set_title(title)
        ax.set_xlabel(r'$\Delta U_i(\vec{x}) / U_i(\vec{x})$')
        ax.set_ylabel(r'Probability density')
        # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

        # ax.legend(loc=1)
        cax, _ = mpl.colorbar.make_axes(ax)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize, label=r'$\Delta t (a.u)$')
for i, fig in enumerate([fig1, fig2, fig3, fig4]):
    graph.suptitle(suptitles[i] + ', $W_{in}= 8W$ px', fignum=i+1)
    imgfilename = '/Volumes/bigraid/takumi/turbulence/JHTD/synthetic_data/hdf5data/tstep1_npt50000_lt20p0_pbcTrue_varyingDt/post_processed/analysisresults_local/' + 'noise_pdf_%d' % (i+1)
    graph.save(imgfilename, fignum=i+1)
    graph.save(imgfilename, fignum=i+1, ext='png')

import matplotlib.pyplot as plt

plt.close('all')

# plot fit results
for ind in range(4):
    fit_data = zip(iw_all[ind], deltat_all[ind], gamma_all[ind], chi_all[ind], fit_err_all[ind], r2s[ind])
    for iw in iws:
        fit_data_W = search_tuple(fit_data, 'W%d' % iw)
        iw_list, deltat_list, gamma_list, chi_list, fit_err_test, r2_list = zip(*fit_data_W)
        if iw > 16:
            indices = [rr > 0.9 for rr in r2_list]
            indices = [True, True, True, True, True, True, False, False, False]
        else:
            indices = [rr > 0.0 for rr in r2_list]
        fig_temp1, ax_temp1, cp = graph.errorfill(np.array(deltat_list)[indices] * dt_sim * dt_spacing / tau, np.abs(np.array(gamma_list)[indices]), yerr=np.array(fit_err_test)[indices][:, 1],
                                                  fignum=5 + ind, subplot=121, label=r'$W=%d$ px' % iw, lw=lw)
        fig_temp1, ax_temp2, cp = graph.errorfill(np.array(deltat_list)[indices] * dt_sim * dt_spacing / tau, np.array(chi_list)[indices], yerr=np.array(fit_err_test)[indices][:, 0],
                                                  fignum=5 + ind, subplot=122, label=r'$W=%d$ px' % iw, lw=lw, figsize=(16, 8))
    ax_temp1.legend()
    ax_temp1_2 = ax_temp1.twiny()
    ax_temp1.set_xlim(0, 1)
    ax_temp2.set_ylim(0, 0.35)
    ax_temp1.set_xlabel(r'Normalized time $ t / \tau$')
    new_tick_locations = np.array([0, 0.05/tau, 0.1/tau, 0.15/tau, 0.2/tau])
    def tick_function(x):
        new_x = x * tau
        return ["%.3f" % z for z in new_x]
    ax_temp1_2.set_xlim(ax_temp1.get_xlim())
    ax_temp1_2.set_xticks(new_tick_locations)
    ax_temp1_2.set_xticklabels(tick_function(new_tick_locations))
    ax_temp1_2.set_xlabel(r'Time t (a.u.)')
    ax_temp1.set_ylabel(r'$\gamma$')


    ax_temp2.legend()
    ax_temp2_2 = ax_temp2.twiny()
    ax_temp2.set_xlim(0, 1)
    # ax_temp2.set_ylim(-0.4, 0.4)
    ax_temp2.set_xlabel(r'Normalized time $ t / \tau$')
    new_tick_locations = np.array([0, 0.05 / tau, 0.1 / tau, 0.15 / tau, 0.2 / tau])

    ax_temp2_2.set_xlim(ax_temp1.get_xlim())
    ax_temp2_2.set_xticks(new_tick_locations)
    ax_temp2_2.set_xticklabels(tick_function(new_tick_locations))
    ax_temp2_2.set_xlabel(r'Time t (a.u.)')
    ax_temp2.set_ylabel(r'$\chi$')

    imgfilename = '/Volumes/bigraid/takumi/turbulence/JHTD/synthetic_data/hdf5data/tstep1_npt50000_lt20p0_pbcTrue_varyingDt/post_processed/analysisresults_local/' + 'fit_results_noise_pdf_%d' % (ind + 1)
    graph.save(imgfilename, fignum=5+ind)
    graph.save(imgfilename, fignum=5+ind, ext='png')



# graph.errorfill(deltat_all[0], gamma_all[0], yerr=np.array(fit_err_all[0])[:, 0], fignum=5)

