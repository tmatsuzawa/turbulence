"""
Load circulation data (middle-sized box) and (small box)
Load veff from piston tracking
Plot circulation as a function of veff etc.
"""
import numpy as np
import pprint
import json
import library.tools.rw_data as rw
import library.basics.formatstring as fs
import library.basics.formatarray as fa
import library.display.graph as graph
import library.basics.std_func as std_func
import glob
import codecs
import os
import matplotlib.pyplot as plt

base_dir = '/Volumes/labshared3-1/takumi/Similar_flows/'
data_PIV_basedir_list = [base_dir+'2018_03_27/singlering/PIV_AnalysisResults/',
                         base_dir+'2018_04_02/singlering/PIV_AnalysisResults/']
data_tracking_basedir_list = [base_dir + '2018_03_27/pistontracking/',
                              base_dir + '2018_04_02/pistontracking/']
# data_PIV_basedir_list = [base_dir+'2018_04_02/singlering/PIV_AnalysisResults/']
# data_tracking_basedir_list = [base_dir + '2018_04_02/pistontracking/']


#data_tracking_dir_list = glob.glob(data_tracking_basedir_list + '*')

def update_data_dict(dict, key, subkey, subsubkey, data):
    """
    Generate a dictionary that stores effective velocity
    Parameters
    ----------
    dict
    key: span like span5.4
    subkey: commanded velocity, str
    data: effective velocity, float

    Returns
    -------

    """
    print key, subkey, subsubkey
    if not key in dict:
        dict[key] = {}  # Generate a sub-dictionary
    if not subkey in dict[key]:
        dict[key][subkey] = {}
    dict[key][subkey][subsubkey] = data
    return dict

def compute_stroke_ratio_LD(span, orifice_d=25.6, piston_d=160.,  num_orifices=1):
    LD = (piston_d / orifice_d)**2 * span / orifice_d / num_orifices
    return LD

def fit_func_for_vring_vs_veff(x, a, b):
    return a*x*(np.log(x)+b)


vr_all, gammar_all, veff_all = [], [], []

for i, data_PIV_basedir in enumerate(data_PIV_basedir_list):
    veff_dict, gammar_dict, gammar_err_dict, actual_span_dict, actual_span_ratio_dict = {}, {}, {}, {}, {}
    vr_dict, vr_err_dict = {}, {}
    print '!!!!!!!!!!!!'
    print data_PIV_basedir
    data_PIV_dir_list = glob.glob(data_PIV_basedir + '*')
    for data_PIV_dir in data_PIV_dir_list:
        # Get stroke length, commanded velocity, and freq from filename
        fn_head, fn_tail = os.path.split(data_PIV_dir)
        span = fs.get_float_from_str(fn_tail, 'piston', 'mm_freq')
        vc =  fs.get_float_from_str(fn_tail, 'Hz_v', 'mms')
        freq = fs.get_float_from_str(fn_tail, 'freq', 'Hz')
        # get some strings from filename
        span_str = fs.get_str_from_str(fn_tail, 'piston', 'mm_freq')
        vc_str =  fs.get_str_from_str(fn_tail, 'Hz_v', 'mms')
        freq_str = fs.get_str_from_str(fn_tail, 'freq', 'Hz')
        # try:
        #     trial_no_str = fs.get_str_from_str(fn_tail, 'trial', '')
        #     print fn_tail
        # except NameError:
        #     trial_no_str = fn_tail[-1]
        #     print fn_tail
        date_str = fs.get_str_from_str(data_PIV_dir, base_dir, '/singlering')
        date_str = date_str.replace('_', '')
        trial_no_str = fn_tail[-1]

        # print span_str, vc_str, freq_str, trial_no_str, date_str
        # print trial_no_str
        # Get circulation data
        circulation_datapath = data_PIV_dir + '/time_evolution/time_evolution_data.json'
        try:
            circulation_data = rw.read_json(circulation_datapath, verbose=False)
        except IOError:
            continue
        for key in circulation_data:
            if key == 'gammar':
                gammar = circulation_data[key]
            if key == 'vr':
                vr = circulation_data[key]


        ## Compute circulation and ring velocity
        gammar_avg = np.mean(gammar)
        gammar_std = np.std(gammar)
        vr_avg = np.mean(vr)
        vr_std = np.std(vr)

        # Check if a corresponding tracking video exists
        tracking_dir = date_str + '_4000fps_span' + span_str + 'mm_v' + vc_str + 'mms_f' + freq_str + 'Hz_setting1_trial' + trial_no_str
        print data_tracking_basedir_list[i]+tracking_dir
        print os.path.isdir(data_tracking_basedir_list[i] + tracking_dir)

        if os.path.isdir(data_tracking_basedir_list[i]+tracking_dir):
            #load veff and stroke length
            veff_datafilepath = data_tracking_basedir_list[i] + tracking_dir + '/plots/veff.pkl'
            veff = rw.read_pickle(veff_datafilepath)
            span_datafilepath = data_tracking_basedir_list[i] + tracking_dir + '/plots/strokelength.pkl'
            actual_span = rw.read_pickle(span_datafilepath)
            actual_span_ratio = actual_span/span  # report ratio
            print actual_span, span, actual_span_ratio

            #update data dict
            key = date_str + 'span' + '%04.1f' % span
            subkey = 'vp_commanded' + str(vc)
            subsubkey = 'trial' + trial_no_str
            if date_str=='20180327':
                alpha = 1.28
            else:
                alpha = 1
            update_data_dict(veff_dict, key, subkey, subsubkey, np.abs(veff))
            update_data_dict(gammar_dict, key, subkey, subsubkey, gammar_avg)
            update_data_dict(gammar_err_dict, key, subkey, subsubkey, gammar_std)
            update_data_dict(actual_span_dict, key, subkey, subsubkey, np.abs(actual_span))
            update_data_dict(actual_span_ratio_dict, key, subkey, subsubkey, np.abs(actual_span_ratio))
            update_data_dict(vr_dict, key, subkey, subsubkey, vr_avg)
            update_data_dict(vr_err_dict, key, subkey, subsubkey, vr_std)

    # print gammar_dict
    # print veff_dict

    for key in sorted(gammar_dict.keys(), reverse=True):
        # Specify keys you would not like to show on the plot
        #skipkeylist = ['20180402span01.3', '20180402span02.0']
        skipkeylist=[]
        if not key in skipkeylist:
            if '20180402' in key:
                fmt = 'v'
                orifice_d = 20.0 # mm small box
                piston_d = 125.0 # mm
            else:
                fmt = 'o'
                orifice_d = 25.6  # mm middle-sized box
                piston_d = 160.0  # mm
            if '20180327' in key:
                alpha = 1.28
            else:
                alpha = 1


            label = key + 'mm'
            span = fs.get_float_from_str(label, 'span', 'mm')
            # Compute L/D value
            LD = compute_stroke_ratio_LD(span, orifice_d=orifice_d, piston_d=piston_d)
            # Use L/D values as labels
            if '20180402' in key:
                label = 'L/D=%.2f (small box)' % LD
            else:
                label = 'L/D=%.2f (mid-sized box)' % LD

            vp_commanded_list, veff_list, gammar_list, gammar_err_list, actual_span_list, = [], [], [], [], []
            actual_span_ratio_list, Lveff, vr_list, vr_err_list = [], [], [], []
            actual_span_ratio_list1 = []

            for subkey in gammar_dict[key]:
                for subsubkey in gammar_dict[key][subkey]:
                    #print key, subkey, subsubkey
                    #vp_commanded_list.append(float(subkey[12:]))
                    veff_list.append(veff_dict[key][subkey][subsubkey])
                    gammar_list.append(gammar_dict[key][subkey][subsubkey])
                    gammar_err_list.append(gammar_err_dict[key][subkey][subsubkey])
                    actual_span_list.append(actual_span_dict[key][subkey][subsubkey])
                    actual_span_ratio_list.append(actual_span_ratio_dict[key][subkey][subsubkey])
                    Lveff.append(actual_span_dict[key][subkey][subsubkey]**(1./3) * veff_dict[key][subkey][subsubkey])
                    vr_list.append(vr_dict[key][subkey][subsubkey])
                    vr_err_list.append(vr_err_dict[key][subkey][subsubkey])
                    # All data points
                    gammar_all.append(gammar_dict[key][subkey][subsubkey])
                    vr_all.append(vr_dict[key][subkey][subsubkey])
                    veff_all.append(veff_dict[key][subkey][subsubkey])

                    #print veff_dict[key][subkey][subsubkey], gammar_dict[key][subkey][subsubkey]
            # Sort the data lists
            # veff and gammar
            veff_list_for_gammar, gammar_list1 = fa.sort_two_arrays_using_order_of_first_array(
                veff_list, gammar_list)
            # L^1/3*veff and gammar
            Lveff_for_gammar, gammar_list2 = fa.sort_two_arrays_using_order_of_first_array(
                Lveff, gammar_list)
            # veff and stroke length ratio
            veff_list_for_actual_span, actual_span_ratio_list1 = fa.sort_two_arrays_using_order_of_first_array(
                veff_list, actual_span_ratio_list)
            # vring and gammar
            vr_list_for_gammar, gammar_list3 = fa.sort_two_arrays_using_order_of_first_array(
                vr_list, gammar_list)
            # vring and vring_err
            vr_list_for_verr, vr_err_list1 = fa.sort_two_arrays_using_order_of_first_array(
                vr_list, vr_err_list)
            # veff and vring
            veff_list_for_vring, vr_list2 = fa.sort_two_arrays_using_order_of_first_array(
                veff_list, vr_list)
            # veff and vring_err
            veff_list_for_vring, vr_err_list2 = fa.sort_two_arrays_using_order_of_first_array(
                veff_list, vr_err_list)

            # Circulation vs veff
            fig1, ax1 = graph.errorbar(veff_list_for_gammar, gammar_list1, yerr=gammar_err_list, fignum=1,
                                   label=label, fmt=fmt)
            graph.labelaxes(ax1, '|$\overline{v^2_p} / \overline{v_p} \alpha $| [mm/s]', '$\Gamma [mm^2/s]$ ')
            ax1.legend()
            graph.setaxes(ax1, 0, 300, 0, 15000)
            graph.save(base_dir + 'gamma_vs_Veff')

            # Stroke length
            fig2, ax2 = graph.plot(veff_list_for_actual_span, actual_span_ratio_list1, fignum=2, marker=fmt,
                                   label=label)
            graph.labelaxes(ax2, '|$\overline{v^2_p} / \overline{v_p}$| [mm/s]', '$L/ L_c[mm^2/s]$ ')
            ax2.legend()
            graph.setaxes(ax2, 0, 300, 0, 1.5)
            graph.axhline(ax2, 1)
            graph.save(base_dir + 'strokelength')

            # Circulation vs L^1/3 veff
            fig3, ax3 = graph.errorbar(Lveff_for_gammar, gammar_list2, yerr=gammar_err_list, fignum=3,
                                   label=label, fmt=fmt)
            graph.labelaxes(ax3, '$L^{1/3}$|$\overline{v^2_p} / \overline{v_p}\alpha$| [mm/s]', '$\Gamma [mm^2/s]$ ')
            graph.setaxes(ax3, 0, 300, 0, 15000)
            ax3.legend()

            graph.save(base_dir+'gamma_vs_LVeff')

            # Circulation vs vring
            fig4, ax4 = graph.errorbar(vr_list_for_gammar, gammar_list3, xerr=vr_err_list1, yerr=gammar_err_list,
                                       fignum=4, label=label, fmt=fmt)
            graph.labelaxes(ax4, '$v_{ring} $ [mm/s]', '$\Gamma [mm^2/s]$ ')
            ax4.legend()
            graph.setaxes(ax4, 0, 300, 0, 15000)

            # Vring vs veff
            fig5, ax5 = graph.errorbar(veff_list_for_vring, vr_list2, yerr=vr_err_list,
                                       fignum=5, label=label, fmt=fmt)
            graph.labelaxes(ax5, '|$\overline{v^2_p} / \overline{v_p}$| [mm/s]', '$v_{ring} $ [mm/s]')
            ax5.legend()
            graph.setaxes(ax5, 0, 300, 0, 300)
            #graph.save(base_dir + 'vring_vs_veff')

            print veff_list
            print vr_list


    # Add a fit curve obtained from last fall
    # x = np.linspace(0, 200, 100)
    # graph.plotfunc(fit_func_for_vring_vs_veff, x, [0.3394, -0.2048], linestyle=':', color='r', fignum=5)
    graph.setaxes(ax5, 0, 300, 0, 300)
    # graph.save(base_dir + 'vring_vs_veff')


    # Add a fit curve
    # fig4, ax4, popt, pcov = graph.plot_fit_curve(vr_all, gammar_all, fignum=4, label='fit', linestyle=':', color='r',
    #                                              xmin = 0, xmax = 300, add_equation=False)
    # text = '$y=ax+b$: a=%.2f, b=%.2f' % (popt[0], popt[1])
    # graph.addtext(ax4, text, x=100, y=2000)

    # Compare to the fit curve obtained in Sep 2017
    # param = [60, 0]
    # x = np.linspace(0, 300, 100)
    # fig4, ax4 = graph.plotfunc(std_func.linear_func, x, param, linestyle=':', color='r', fignum=4)

    graph.save(base_dir + 'gamma_vs_Vring')


    output_dict = {}
    output_dict["veff_all"] = veff_all
    output_dict["vr_all"] = vr_all
    output_dict["gammar_all"] = gammar_all
    output_path = base_dir + 'output_032718_040218.json'
    rw.write_json(output_path, output_dict)


plt.show()










