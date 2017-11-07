# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:27:38 2015
test
"""

'''measurement of the taylor scale'''

import numpy as np
# import turbulence.analysis.strain_tensor as strain_tensor
import turbulence.display.graphes as graphes
import matplotlib.pyplot as plt

def compute(M, i, Dt=50, display=False):
    # compute the taylor scale by averaging U dU/dx over space.
    # the derivative can be taken either in the direction of U or in the direction perpendicular to it
    # call functions from the derivative module, that compute spatial derivative accurately
    # author: stephane
    nx, ny, nt = M.shape()

    start = max(0, i - Dt // 2)
    end = min(nt, i + Dt // 2)

    n = end - start
    Ux = M.Ux[:, :, start:end]
    Uy = M.Uy[:, :, start:end]

    # compute the strain tensor from Ux and Uy components
    edge = 3;
    d = 2
    dU = np.zeros((nx - edge * 2, ny - edge * 2, d, d, n))

    fx = max([np.mean(np.diff(M.x)), np.mean(np.diff(M.x))])  # in mm/box

    for k in range(n):
        U = np.transpose(np.asarray([Ux[..., k], Uy[..., k]]),
                         (1, 2, 0))  # shift the dimension to compute the strain tensor along axis 0 and 1
        dU[..., k] = fx * strain_tensor.strain_tensor(U, d=2, step=1)  # strain tensor computed at the box size

    # naive length scale, computed from Ux dUx/dx
    index = (slice(3, -3, None), slice(3, -3, None), slice(None))

    E_dE = Ux[index] * dU[..., 0, 0, :] + Uy[index] * dU[..., 1, 1, :]
    E = np.power(Ux[index], 2) + np.power(Uy[index], 2)

    if display:
        graphes.hist(E_dE / np.std(E_dE), num=1000, label='ko--', fignum=1)
        graphes.hist(E / np.std(E), num=1000, label='r^-', fignum=1)
        graphes.set_axes(-10, 10, 1, 10 ** 5)
        graphes.legende('E', 'pdf(E)', '')

    lambda_R0 = np.mean(E) / np.std(E_dE)
    print('')
    print(str(M.t[i]) + ' : ' + str(lambda_R0))
    #    input()

    dtheta = np.pi / 100
    angles = np.arange(0, np.pi, dtheta)

    E_dE_l = []
    E_dE_t = []
    E_theta = []

    lambda_R_l = []
    lambda_R_t = []

    for j, theta in enumerate(angles):
        U_theta = Ux[index] * np.cos(theta) + Uy[index] * np.sin(theta)

        dU_l = dU[..., 0, 0, :] * np.cos(theta) + dU[..., 1, 1, :] * np.sin(theta)
        dU_t = dU[..., 1, 0, :] * np.cos(theta) + dU[..., 0, 1, :] * np.sin(
            theta)  # derivative of the same component, but in the normal direction

        # longitudinal of U dU
        E_dE_l.append(np.std(U_theta * dU_l))
        E_dE_t.append(np.std(U_theta * dU_t))
        E_theta.append(np.mean(np.power(U_theta, 2)))

        lambda_R_l.append(E_theta[j] / E_dE_l[j])
        lambda_R_t.append(E_theta[j] / E_dE_t[j])

    lambda_Rl = np.mean(np.asarray(lambda_R_l))
    lambda_Rt = np.mean(np.asarray(lambda_R_t))

    lambda_Rl_std = np.std(np.asarray(lambda_R_l))
    lambda_Rt_std = np.std(np.asarray(lambda_R_t))

    print(str(M.t[i]) + ' : ' + str(lambda_Rl))
    print(str(M.t[i]) + ' : ' + str(lambda_Rt))

    #    graphes.graph(angles,E_dE_l,fignum=1,label='ko')
    #    graphes.graph(angles,E_dE_t,fignum=1,label='r^')

    #    lambda_R = lambda_Rl
    lambdas = {}
    lambdas['l_moy'] = lambda_Rl
    lambdas['t_moy'] = lambda_R0
    lambdas['l_std'] = lambda_Rl_std
    lambdas['t_std'] = lambda_Rt_std

    Urms = np.sqrt(np.std(E))  # E is in mm^2/s^-2
    return lambdas, Urms


def taylor_scale(M, fignum=1, display=True, label='k^'):
    #author: stephane
    nx, ny, nt = M.shape()
    t = M.t
    Dt = 20
    step = 1

    lambda_R = {}
    Urms = []
    t_R = []
    for i in range(Dt, nt - Dt, step):
        t_R.append(t[i])
        lambdas, U = compute(M, i, Dt=Dt)
        Urms.append(U)
        if lambda_R == {}:
            for key in lambdas.keys():
                lambda_R[key] = [lambdas[key]]
        else:
            for key in lambdas.keys():
                lambda_R[key] += [lambdas[key]]

    graphes.semilogx(t_R, lambda_R['t_moy'], fignum=fignum, label=label[0] + '^')
    graphes.semilogx(t_R, lambda_R['l_moy'], fignum=fignum, label=label[0] + '>')

    graphes.graphloglog(t_R, np.asarray(Urms) * np.asarray(lambda_R['t_moy']), fignum=fignum + 1, label=label)
    graphes.graphloglog(np.asarray(Urms), np.asarray(lambda_R['t_moy']), fignum=fignum + 2, label=label)
    graphes.legende('<U''>', 'lambda', '')


def compute_u2_duidxi(Mfluc):
    """
    Returns time averaged U2, Ux2, Uy2, dUidxi2, dUxdx2, dUydy2 at every point
    Averaging is done manually because Mfluc may contain inf and nan simultaneously.
    -> This issue can be better handled by using numpy masking modules; but this code simply averages
     without using the masking modules.
    Parameters
    ----------
    Mfluc: M class object

    Returns
    -------

    """

    Mfluc.Ux2 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1], Mfluc.Ux.shape[2]))
    Mfluc.Uy2 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1], Mfluc.Ux.shape[2]))
    Mfluc.U2 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1], Mfluc.Ux.shape[2]))

    Mfluc.dUxdx = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1], Mfluc.Ux.shape[2]))
    Mfluc.dUydy = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1], Mfluc.Ux.shape[2]))

    Mfluc.dUidxi2 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1], Mfluc.Ux.shape[2]))
    Mfluc.dUxdx2 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1], Mfluc.Ux.shape[2]))
    Mfluc.dUydy2 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1], Mfluc.Ux.shape[2]))

    Mfluc.U2ave = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    Mfluc.Ux2ave = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    Mfluc.Uy2ave = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))

    Mfluc.dUidxi2ave = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    Mfluc.dUxdx2ave = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    Mfluc.dUydy2ave = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))


    nu = 1.004  # [mm^2/s]: kinematic viscosity of water at 20C
    print Mfluc.Ux.shape

    tmin = 0
    Dx = np.absolute(np.nanmean(np.diff(Mfluc.x[0, ...])))
    Dy = np.absolute(np.nanmean(np.diff(Mfluc.y[..., 0])))

    for t in range(tmin, Mfluc.Ux.shape[2]):
        for x in range(0, Mfluc.Ux.shape[0] - 1):
            for y in range(0, Mfluc.Ux.shape[1] - 1):
                if x == Mfluc.Ux.shape[0] - 1 or y == Mfluc.Ux.shape[1] - 1:
                    continue

                Mfluc.Ux2[x, y, t] = Mfluc.Ux[x, y, t] * Mfluc.Ux[x, y, t]
                Mfluc.Uy2[x, y, t] = Mfluc.Uy[x, y, t] * Mfluc.Uy[x, y, t]
                Mfluc.dUxdx[x, y, t] = (Mfluc.Ux[x + 1, y, t] - Mfluc.Ux[x, y, t]) / Dx  # du'_x/dx
                Mfluc.dUydy[x, y, t] = (Mfluc.Uy[x, y + 1, t] - Mfluc.Uy[x, y, t]) / Dy  # du'_y/dy
                Mfluc.dUxdx2[x, y, t] = Mfluc.dUxdx[x, y, t] * Mfluc.dUxdx[x, y, t]
                Mfluc.dUydy2[x, y, t] = Mfluc.dUydy[x, y, t] * Mfluc.dUydy[x, y, t]

                Mfluc.U2[x, y, t] = (Mfluc.Ux2[x, y, t] + Mfluc.Uy2[x, y, t])
                Mfluc.dUidxi2[x, y, t] = Mfluc.dUxdx[x, y, t] * Mfluc.dUxdx[x, y, t] + Mfluc.dUydy[x, y, t] * \
                                                                                       Mfluc.dUydy[x, y, t]

    ##Time average of u^2 and (du/dx)^2
    ## np.nanmean may not output proper mean when inf&nan are contained in the arrays
    # Mfluc.U2ave = np.nanmean(Mfluc.U2[...,indices],axis=2)
    # Mfluc.dUidxi2ave = np.nanmean(Mfluc.dUidxi2[...,indices],axis=2)

    counter_1 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    counter_2 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    counter_3 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    counter_4 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    counter_5 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    counter_6 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))

    for t in range(tmin, Mfluc.Ux.shape[2]):
        for x in range(0, Mfluc.Ux.shape[0]):
            for y in range(0, Mfluc.Ux.shape[1]):
                #             if np.isnan(Mfluc.U2[x,y,t])==False and np.isinf(Mfluc.U2[x,y,t])==False:
                #                 Mfluc.U2ave[x,y] += Mfluc.U2[x,y,t]
                #                 counter_1[x,y]+=1
                if np.isnan(Mfluc.Ux2[x, y, t]) == False and np.isinf(Mfluc.Ux2[x, y, t]) == False:
                    Mfluc.Ux2ave[x, y] += Mfluc.Ux2[x, y, t]
                    counter_2[x, y] += 1
                if np.isnan(Mfluc.Uy2[x, y, t]) == False and np.isinf(Mfluc.Uy2[x, y, t]) == False:
                    Mfluc.Uy2ave[x, y] += Mfluc.Uy2[x, y, t]
                    counter_3[x, y] += 1

                    #             if np.isnan(Mfluc.dUidxi2[x,y,t])==False and np.isinf(Mfluc.dUidxi2[x,y,t])==False:
                    #                 counter_4[x,y]+=1
                    #                 Mfluc.dUidxi2ave[x,y] += Mfluc.dUidxi2[x,y,t]
                if np.isnan(Mfluc.dUxdx2[x, y, t]) == False and np.isinf(Mfluc.dUxdx2[x, y, t]) == False:
                    Mfluc.dUxdx2ave[x, y] += Mfluc.dUxdx2[x, y, t]
                    counter_5[x, y] += 1
                if np.isnan(Mfluc.dUydy2[x, y, t]) == False and np.isinf(Mfluc.dUydy2[x, y, t]) == False:
                    Mfluc.dUydy2ave[x, y] += Mfluc.dUydy2[x, y, t]
                    counter_6[x, y] += 1

    print ('Calculating the mean U2 and dUidxi2...')

    for x in range(0, Mfluc.Ux.shape[0]):
        for y in range(0, Mfluc.Ux.shape[1]):
            #         if counter_1[x,y]==0:
            #             Mfluc.U2ave[x,y] = 0
            #         else:
            #             Mfluc.U2ave[x,y] = Mfluc.U2ave[x,y]/counter_1[x,y]
            if counter_2[x, y] == 0:
                Mfluc.Ux2ave[x, y] = 0
            else:
                Mfluc.Ux2ave[x, y] = Mfluc.Ux2ave[x, y] / counter_2[x, y]
            if counter_3[x, y] == 0:
                Mfluc.Uy2ave[x, y] = 0
            else:
                Mfluc.Uy2ave[x, y] = Mfluc.Uy2ave[x, y] / counter_3[x, y]

                #         if counter_4[x,y]==0:
                #             Mfluc.dUidxi2ave[x,y] = 0
                #         else:
                #             Mfluc.dUidxi2ave[x,y] = Mfluc.dUidxi2ave[x,y]/counter_4[x,y]
            if counter_5[x, y] == 0:
                Mfluc.dUxdx2ave[x, y] = 0
            else:
                Mfluc.dUxdx2ave[x, y] = Mfluc.dUxdx2ave[x, y] / counter_5[x, y]
            if counter_6[x, y] == 0:
                Mfluc.dUydy2ave[x, y] = 0
            else:
                Mfluc.dUydy2ave[x, y] = Mfluc.dUydy2ave[x, y] / counter_6[x, y]

    Mfluc.U2ave[...] = (Mfluc.Ux2ave[...] + Mfluc.Uy2ave[...]) / 2
    Mfluc.dUidxi2ave[...] = (Mfluc.dUxdx2ave[...] + Mfluc.dUydy2ave[...]) / 2

    print 'Done'

    #return Mfluc.U2ave, Mfluc.Ux2ave, Mfluc.Uy2ave,  Mfluc.dUidxi2ave, Mfluc.dUxdx2ave, Mfluc.dUydy2ave
    return Mfluc

def plot_lambda_Re_lambda_heatmaps(Mfluc,cutoff=10**(-3),vminRe=0,vmaxRe=800, vminLambda=0,vmaxLambda=800, nu=1.004):
    Mfluc.lambdaT_Local = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    Mfluc.lambdaTx_Local = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    Mfluc.lambdaTy_Local = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    Mfluc.Re_lambdaT_Local = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    Mfluc.Re_lambdaTx_Local = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    Mfluc.Re_lambdaTy_Local = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))

    try :
        Mfluc.U2ave
    except AttributeError:
        print 'Mfluc does not have attribute Mfluc.U2ave. Calculate it.'
        Mfluc=compute_u2_duidxi(Mfluc)

    # Calculate Local Taylor microscale: lambdaT
    for x in range(0, Mfluc.Ux.shape[0] - 1):
        for y in range(0, Mfluc.Ux.shape[1] - 1):
            if Mfluc.dUidxi2ave[x, y] < cutoff:
                Mfluc.lambdaT_Local[x, y] = 0
            else:
                Mfluc.lambdaT_Local[x, y] = np.sqrt(Mfluc.U2ave[x, y] / Mfluc.dUidxi2ave[x, y])
                Mfluc.Re_lambdaT_Local[x, y] = Mfluc.lambdaT_Local[x, y] * np.sqrt(Mfluc.U2ave[x, y]) / nu
            if Mfluc.dUxdx2ave[x, y] < cutoff:
                Mfluc.lambdaTx_Local[x, y] = 0
            else:
                Mfluc.lambdaTx_Local[x, y] = np.sqrt(Mfluc.Ux2ave[x, y] / Mfluc.dUxdx2ave[x, y])
                Mfluc.Re_lambdaTx_Local[x, y] = Mfluc.lambdaTx_Local[x, y] * np.sqrt(Mfluc.Ux2ave[x, y]) / nu
            if Mfluc.dUydy2ave[x, y] < cutoff:
                Mfluc.lambdaTy_Local[x, y] = 0
            else:
                Mfluc.lambdaTy_Local[x, y] = np.sqrt(Mfluc.Uy2ave[x, y] / Mfluc.dUydy2ave[x, y])
                Mfluc.Re_lambdaTy_Local[x, y] = Mfluc.lambdaTy_Local[x, y] * np.sqrt(Mfluc.Uy2ave[x, y]) / nu

                # Plot Taylor miroscale

    graphes.color_plot(Mfluc.x, Mfluc.y, Mfluc.Re_lambdaT_Local, fignum=1, vmin=vminRe, vmax=vmaxRe)
    graphes.colorbar()
    plt.title('Local $Re_\lambda$')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    graphes.color_plot(Mfluc.x, Mfluc.y, Mfluc.Re_lambdaTx_Local, fignum=2, vmin=vminRe, vmax=vmaxRe)
    graphes.colorbar()
    plt.title('Local $Re_{\lambda,x}$')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    graphes.color_plot(Mfluc.x, Mfluc.y, Mfluc.Re_lambdaTy_Local, fignum=3, vmin=vminRe, vmax=vmaxRe)
    graphes.colorbar()
    plt.title('Local $Re_{\lambda,y}$')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')

    graphes.color_plot(Mfluc.x, Mfluc.y, Mfluc.lambdaT_Local, fignum=4, vmin=vminLambda, vmax=vmaxLambda)
    graphes.colorbar()
    plt.title('Local $\lambda$')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    graphes.color_plot(Mfluc.x, Mfluc.y, Mfluc.lambdaTx_Local, fignum=5, vmin=vminLambda, vmax=vmaxLambda)
    graphes.colorbar()
    plt.title('Local $\lambda_x$')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    graphes.color_plot(Mfluc.x, Mfluc.y, Mfluc.lambdaTy_Local, fignum=6, vmin=vminLambda, vmax=vmaxLambda)
    graphes.colorbar()
    plt.title('Local $\lambda_y$')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')

def compute_lambda_Re_lambda(Mfluc, nu=1.004):
    try :
        Mfluc.U2ave  # Check if  Mfluc has attribute called U2ave
    except AttributeError:
        print 'Mfluc does not have attribute Mfluc.U2ave. Calculate it.'
        Mfluc=compute_u2_duidxi(Mfluc)

    Mfluc.lambdaT = np.sqrt(np.nanmean(Mfluc.U2ave)/np.nanmean(Mfluc.dUidxi2ave))
    Mfluc.lambdaTx = np.sqrt(np.nanmean(Mfluc.Ux2ave)/np.nanmean(Mfluc.dUxdx2ave))
    Mfluc.lambdaTy = np.sqrt(np.nanmean(Mfluc.Uy2ave)/np.nanmean(Mfluc.dUydy2ave))

    Mfluc.Re_lambdaT = np.sqrt(1.0 / 2.0 * np.nanmean(Mfluc.U2ave)) * Mfluc.lambdaT / nu
    Mfluc.Re_lambdaTx = np.sqrt(np.nanmean(Mfluc.Ux2ave)) * Mfluc.lambdaTx / nu
    Mfluc.Re_lambdaTy = np.sqrt(np.nanmean(Mfluc.Uy2ave)) * Mfluc.lambdaTy / nu
    print 'lambdaT, Re_lambdaT, lambdaTx, Re_lambdaTx, lambdaTy, Re_lambdaTy'
    print Mfluc.lambdaT, Mfluc.Re_lambdaT, Mfluc.lambdaTx, Mfluc.Re_lambdaTx, Mfluc.lambdaTy, Mfluc.Re_lambdaTy
    return Mfluc