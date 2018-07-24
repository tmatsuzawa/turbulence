# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:27:38 2015
test
"""

'''measurement of the taylor scale'''

import numpy as np
import turbulence.jhtd.strain_tensor as strain_tensor
import turbulence.display.graphes as graphes
import matplotlib.pyplot as plt

# compute and taylor_scale is not very clear what they do... - takumi


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
        plt.ylabel('Normalized E_dE / np.std(E_dE)')
        plt.xlabel('bins')
        graphes.hist(E / np.std(E), num=1000, label='r^-', fignum=1)
        plt.ylabel('E / np.std(E)')
        plt.xlabel('bins')
        # graphes.set_axis(0, 10, 1, 10 ** 5)
        # graphes.legende('E', 'pdf(E)', '')

    lambda_R0 = np.mean(E) / np.std(E_dE)
    print('lambda_R0, lambda_Rl, lambda_Rt')
    print('t = ' + str(M.t[i]) + ' : ' + str(lambda_R0))

    dtheta = np.pi / 100
    angles = np.arange(0, np.pi, dtheta)

    E_dE_l = []
    E_dE_t = []
    E_theta = []

    lambda_R_l = []
    lambda_R_t = []

    for j, theta in enumerate(angles):
        U_theta = Ux[index] * np.cos(theta) + Uy[index] * np.sin(theta)

        # derivative of the same component, but in the normal direction
        dU_l = dU[..., 0, 0, :] * np.cos(theta) + dU[..., 1, 1, :] * np.sin(theta)
        dU_t = dU[..., 1, 0, :] * np.cos(theta) + dU[..., 0, 1, :] * np.sin(theta)


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

    print('t = ' + str(M.t[i]) + ' : ' + str(lambda_Rl))
    print('t = ' + str(M.t[i]) + ' : ' + str(lambda_Rt))

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


def compute_u2_duidxi(Mfluc, nu=1.004, x0=None, y0=None, x1=None, y1=None, clean=True):
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
    Mfluc: Mdata object
        new attributes: dUidxi, dUxdx, dUydy, dUidxi2, dUxdx2, dUydy2 (3D array)
                        dUidxi2ave, dUxdx2ave, dUydy2ave (2D array)

    """
    if x0 is None:
        x0, y0 = 0, 0
        x1, y1 = Mfluc.Ux.shape[1], Mfluc.Ux.shape[0]
    print 'Will average from (%d, %d) to (%d, %d)' % (x0, y0, x1, y1)

    # Mfluc.Ux2 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1], Mfluc.Ux.shape[2]))
    # Mfluc.Uy2 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1], Mfluc.Ux.shape[2]))
    # Mfluc.U2 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1], Mfluc.Ux.shape[2]))
    #
    # Mfluc.dUxdx = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1], Mfluc.Ux.shape[2]))
    # Mfluc.dUydy = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1], Mfluc.Ux.shape[2]))
    #
    # Mfluc.dUidxi2 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1], Mfluc.Ux.shape[2]))
    # Mfluc.dUxdx2 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1], Mfluc.Ux.shape[2]))
    # Mfluc.dUydy2 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1], Mfluc.Ux.shape[2]))
    #
    # Mfluc.U2ave = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    # Mfluc.Ux2ave = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    # Mfluc.Uy2ave = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    #
    # Mfluc.dUidxi2ave = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    # Mfluc.dUxdx2ave = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    # Mfluc.dUydy2ave = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))


    Dx = np.absolute(np.nanmean(np.diff(Mfluc.x[0, ...])))  # Dx and Dy should be constants
    Dy = np.absolute(np.nanmean(np.diff(Mfluc.y[..., 0])))  # Dx and Dy should be constants

    # for t in range(tmin, Mfluc.Ux.shape[2]):
    #     for y in range(0, Mfluc.Ux.shape[0] - 1):
    #         for x in range(0, Mfluc.Ux.shape[1] - 1):
    #
    #             Mfluc.Ux2[y, x, t] = Mfluc.Ux[y, x, t] ** 2
    #             Mfluc.Uy2[y, x, t] = Mfluc.Uy[y, x, t] ** 2
    #             Mfluc.dUxdx[y, x, t] = (Mfluc.Ux[y + 1, x, t] - Mfluc.Ux[y, x, t]) / Dx  # du'_x/dx
    #             Mfluc.dUydy[y, x, t] = (Mfluc.Uy[y, x + 1, t] - Mfluc.Uy[y, x, t]) / Dy  # du'_y/dy
    #             Mfluc.dUxdx2[y, x, t] = Mfluc.dUxdx[y, x, t] ** 2
    #             Mfluc.dUydy2[y, x, t] = Mfluc.dUydy[y, x, t] ** 2
    #
    #             Mfluc.U2[y, x, t] = (Mfluc.Ux2[y, x, t] + Mfluc.Uy2[y, x, t])
    #             Mfluc.dUidxi2[y, x, t] = Mfluc.dUxdx[y, x, t] ** 2 + Mfluc.dUydy[y, x, t] ** 2
    Mfluc.Ux2 = Mfluc.Ux ** 2
    Mfluc.Uy2 = Mfluc.Uy ** 2
    Mfluc.dUxdx = np.gradient(Mfluc.Ux, Dx, axis=1)
    Mfluc.dUydy = np.gradient(Mfluc.Uy, Dy, axis=0)
    Mfluc.dUxdx2 = Mfluc.dUxdx ** 2
    Mfluc.dUydy2 = Mfluc.dUydy ** 2

    Mfluc.U2 = (Mfluc.Ux2 + Mfluc.Uy2)
    Mfluc.dUidxi2 = Mfluc.dUxdx ** 2 + Mfluc.dUydy ** 2

    # time average
    ## if data contains
    if clean:
        Mfluc.U2ave = np.nanmean(Mfluc.U2[...], axis=2)
        Mfluc.Ux2ave = np.nanmean(Mfluc.Ux2[...], axis=2)
        Mfluc.Uy2ave = np.nanmean(Mfluc.Uy2[...], axis=2)
        Mfluc.dUidxi2ave = np.nanmean(Mfluc.dUidxi2[...], axis=2)
        Mfluc.dUxdx2ave = np.nanmean(Mfluc.dUxdx2[...], axis=2)
        Mfluc.dUydy2ave = np.nanmean(Mfluc.dUydy2[...], axis=2)
    else:
        # Depreciated. Manually averaging quantities
        counter_1 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1])) # counter for Mfluc.U2
        counter_2 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1])) # counter for Mfluc.Ux2
        counter_3 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1])) # counter for Mfluc.Uy2
        counter_4 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1])) # counter for Mfluc.dUidxi2
        counter_5 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1])) # counter for Mfluc.dUxdx2
        counter_6 = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1])) # counter for Mfluc.dUydy2

        for t in range(tmin, Mfluc.Ux.shape[2]):
            for y in range(y0, y1):
                for x in range(x0, x1):
                    if np.isnan(Mfluc.Ux2[y, x, t]) == False and np.isinf(Mfluc.Ux2[y, x, t]) == False:
                        Mfluc.Ux2ave[y, x] += Mfluc.Ux2[y, x, t]
                        counter_2[y, x] += 1
                    if np.isnan(Mfluc.Uy2[y, x, t]) == False and np.isinf(Mfluc.Uy2[y, x, t]) == False:
                        Mfluc.Uy2ave[y, x] += Mfluc.Uy2[y, x, t]
                        counter_3[y, x] += 1
                    if np.isnan(Mfluc.dUxdx2[y, x, t]) == False and np.isinf(Mfluc.dUxdx2[y, x, t]) == False:
                        Mfluc.dUxdx2ave[y, x] += Mfluc.dUxdx2[y, x, t]
                        counter_5[y, x] += 1
                    if np.isnan(Mfluc.dUydy2[y, x, t]) == False and np.isinf(Mfluc.dUydy2[y, x, t]) == False:
                        Mfluc.dUydy2ave[y, x] += Mfluc.dUydy2[y, x, t]
                        counter_6[y, x] += 1

        print ('Calculating the mean U2 and dUidxi2...')

        for y in range(y0, y1):
            for x in range(x0, x1):
                if counter_2[y, x] == 0:
                    Mfluc.Ux2ave[y, x] = 0
                else:
                    Mfluc.Ux2ave[y, x] = Mfluc.Ux2ave[y, x] / counter_2[y, x]
                if counter_3[y, x] == 0:
                    Mfluc.Uy2ave[y, x] = 0
                else:
                    Mfluc.Uy2ave[y, x] = Mfluc.Uy2ave[y, x] / counter_3[y, x]
                if counter_5[y, x] == 0:
                    Mfluc.dUxdx2ave[y, x] = 0
                else:
                    Mfluc.dUxdx2ave[y, x] = Mfluc.dUxdx2ave[y, x] / counter_5[y, x]
                if counter_6[y, x] == 0:
                    Mfluc.dUydy2ave[y, x] = 0
                else:
                    Mfluc.dUydy2ave[y, x] = Mfluc.dUydy2ave[y, x] / counter_6[y, x]

    Mfluc.U2ave[...] = (Mfluc.Ux2ave[...] + Mfluc.Uy2ave[...]) / 2
    Mfluc.dUidxi2ave[...] = (Mfluc.dUxdx2ave[...] + Mfluc.dUydy2ave[...]) / 2

    print 'Done'

    #return Mfluc.U2ave, Mfluc.Ux2ave, Mfluc.Uy2ave,  Mfluc.dUidxi2ave, Mfluc.dUxdx2ave, Mfluc.dUydy2ave
    return Mfluc

def compute_rate_of_strain_tensor(M):
    x0, y0 = 0, 0
    x1, y1 = M.Ux.shape[1], M.Ux.shape[0]
    Dx = np.absolute(np.nanmean(np.diff(M.x[0, ...])))  # Dx and Dy should be constants
    Dy = np.absolute(np.nanmean(np.diff(M.y[..., 0])))  # Dx and Dy should be constants
    # actual rate of strain tensor
    M.dUxdx = np.gradient(M.Ux, Dx, axis=1)
    M.dUydy = np.gradient(M.Uy, Dy, axis=0)
    M.dUxdy = np.gradient(M.Ux, Dy, axis=0)
    M.dUydx = np.gradient(M.Uy, Dx, axis=1)
    M.Sxx, M.Syy = M.dUxdx, M.dUydy
    M.Sxy = M.Syx = 1. / 2. * (M.dUxdy + M.dUydx) # rate-of-strain tensor is symmetric

    # mean flow
    M.Ux_mean = np.nanmean(M.Ux, axis=2) #mean flow
    M.Uy_mean = np.nanmean(M.Uy, axis=2) #mean flow
    M.dUxdx_mean = np.gradient(M.Ux_mean, Dx, axis=1) #mean flow
    M.dUxdy_mean = np.gradient(M.Ux_mean, Dy, axis=0) #mean flow
    M.dUydx_mean = np.gradient(M.Uy_mean, Dx, axis=1) #mean flow
    M.dUydy_mean = np.gradient(M.Uy_mean, Dy, axis=0) #mean flow
    M.Sxx_mean, M.Syy_mean = M.dUxdx_mean, M.dUydy_mean
    M.Sxy_mean = M.Syx_mean = 1. / 2. * (M.dUxdy_mean + M.dUydx_mean) # rate-of-strain tensor is symmetric


    # fluctuating rate of strain tensor
    Ux_fluc, Uy_fluc = np.zeros(M.Ux.shape), np.zeros(M.Ux.shape)
    for t in range(M.Ux.shape[2]):
        Ux_fluc[..., t], Uy_fluc[..., t] = M.Ux[..., t] - np.nanmean(M.Ux, axis=2), M.Uy[..., t] - np.nanmean(M.Uy, axis=2)
    M.dUxdx_fluc = np.gradient(Ux_fluc, Dx, axis=1)
    M.dUydy_fluc = np.gradient(Uy_fluc, Dy, axis=0)
    M.dUxdy_fluc = np.gradient(Ux_fluc, Dy, axis=0)
    M.dUydx_fluc = np.gradient(Uy_fluc, Dx, axis=1)
    M.Sxx_fluc, M.Syy_fluc = M.dUxdx_fluc, M.dUydy_fluc
    M.Sxy_fluc = M.Syx_fluc = 1. / 2. * (M.dUxdy_fluc + M.dUydx_fluc) # rate-of-strain tensor is symmetric
    return M


def plot_lambda_Re_lambda_heatmaps(Mfluc,cutoff=10**(-3),vminRe=0,vmaxRe=800, vminLambda=0,vmaxLambda=800, nu=1.004,
                                   x0=None, y0=None, x1=None, y1=None, redo=False):
    """
    Plots a heatmap of local Re_lambda and lambda
    Parameters
    ----------
    Mfluc
    cutoff
    vminRe
    vmaxRe
    vminLambda
    vmaxLambda
    nu

    Returns
    -------

    """
    if x0 is None:
        x0, y0 = 0, 0
        x1, y1 = Mfluc.Ux.shape[1], Mfluc.Ux.shape[0]
        print 'Will average from (%d, %d) to (%d, %d)  (index)' % (x0, y0, x1, y1)
    nrows, ncolumns = y1 - y0, x1 - x0

    Mfluc.lambdaT_Local = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    Mfluc.lambdaTx_Local = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    Mfluc.lambdaTy_Local = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    Mfluc.Re_lambdaT_Local = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    Mfluc.Re_lambdaTx_Local = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    Mfluc.Re_lambdaTy_Local = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))

    if not hasattr(Mfluc, 'U2ave') or redo:
        print 'Mfluc does not have attribute Mfluc.U2ave. Calculate it.'
        Mfluc = compute_u2_duidxi(Mfluc, x0=x0, y0=y0, x1=x1, y1=y1)
        print Mfluc.Ux2ave.shape

    # Calculate Local Taylor microscale: lambdaT
    for y in range(0, Mfluc.Ux.shape[0] - 1):
        for x in range(0, Mfluc.Ux.shape[1] - 1):
            if Mfluc.dUidxi2ave[y, x] < cutoff:
                Mfluc.lambdaT_Local[y, x] = 0
            else:
                Mfluc.lambdaT_Local[y, x] = np.sqrt(Mfluc.U2ave[y, x] / Mfluc.dUidxi2ave[y, x])
                Mfluc.Re_lambdaT_Local[y, x] = Mfluc.lambdaT_Local[y, x] * np.sqrt(Mfluc.U2ave[y, x]) / nu
            if Mfluc.dUxdx2ave[y, x] < cutoff:
                Mfluc.lambdaTx_Local[y, x] = 0
            else:
                Mfluc.lambdaTx_Local[y, x] = np.sqrt(Mfluc.Ux2ave[y, x] / Mfluc.dUxdx2ave[y, x])
                Mfluc.Re_lambdaTx_Local[y, x] = Mfluc.lambdaTx_Local[y, x] * np.sqrt(Mfluc.Ux2ave[y, x]) / nu
            if Mfluc.dUydy2ave[y, x] < cutoff:
                Mfluc.lambdaTy_Local[y, x] = 0
            else:
                Mfluc.lambdaTy_Local[y, x] = np.sqrt(Mfluc.Uy2ave[y, x] / Mfluc.dUydy2ave[y, x])
                Mfluc.Re_lambdaTy_Local[y, x] = Mfluc.lambdaTy_Local[y, x] * np.sqrt(Mfluc.Uy2ave[y, x]) / nu

    # Plot Taylor miroscale
    graphes.color_plot(Mfluc.x[y0:y1, x0:x1], Mfluc.y[y0:y1, x0:x1], Mfluc.Re_lambdaT_Local[y0:y1, x0:x1], fignum=1, vmin=vminRe, vmax=vmaxRe)
    graphes.colorbar()
    plt.title('Local $Re_\lambda$')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    graphes.color_plot(Mfluc.x[y0:y1, x0:x1], Mfluc.y[y0:y1, x0:x1], Mfluc.Re_lambdaTx_Local[y0:y1, x0:x1], fignum=2, vmin=vminRe, vmax=vmaxRe)
    graphes.colorbar()
    plt.title('Local $Re_{\lambda,x}$')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    graphes.color_plot(Mfluc.x[y0:y1, x0:x1], Mfluc.y[y0:y1, x0:x1], Mfluc.Re_lambdaTy_Local[y0:y1, x0:x1], fignum=3, vmin=vminRe, vmax=vmaxRe)
    graphes.colorbar()
    plt.title('Local $Re_{\lambda,y}$')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')

    graphes.color_plot(Mfluc.x[y0:y1, x0:x1], Mfluc.y[y0:y1, x0:x1], Mfluc.lambdaT_Local[y0:y1, x0:x1], fignum=4, vmin=vminLambda, vmax=vmaxLambda)
    graphes.colorbar()
    plt.title('Local $\lambda$')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    graphes.color_plot(Mfluc.x[y0:y1, x0:x1], Mfluc.y[y0:y1, x0:x1], Mfluc.lambdaTx_Local[y0:y1, x0:x1], fignum=5, vmin=vminLambda, vmax=vmaxLambda)
    graphes.colorbar()
    plt.title('Local $\lambda_x$')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    graphes.color_plot(Mfluc.x[y0:y1, x0:x1], Mfluc.y[y0:y1, x0:x1], Mfluc.lambdaTy_Local[y0:y1, x0:x1], fignum=6, vmin=vminLambda, vmax=vmaxLambda)
    graphes.colorbar()
    plt.title('Local $\lambda_y$')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')


def plot_lambda_Re_lambda_coarse(Mfluc, cutoff=10**(-3), vminRe=0,vmaxRe=800, vminLambda=0, vmaxLambda=10, nu=1.004,
                                 x0=None, y0=None, x1=None, y1=None, redo=False):

    if not hasattr(Mfluc, 'U2ave') or redo:
        print 'Mfluc does not have attribute Mfluc.U2ave. Calculate it.'
        Mfluc=compute_u2_duidxi(Mfluc, x0=x0, y0=y0, x1=x1, y1=y1)



    # Calculate Local Taylor microscale: lambdaT
    Mfluc.lambdaT_Local = np.sqrt(Mfluc.U2ave / Mfluc.dUidxi2ave)
    Mfluc.Re_lambdaT_Local = Mfluc.lambdaT_Local * np.sqrt(Mfluc.U2ave) / nu

    Mfluc.lambdaTx_Local = np.sqrt(Mfluc.Ux2ave / Mfluc.dUxdx2ave)
    Mfluc.Re_lambdaTx_Local = Mfluc.lambdaTx_Local * np.sqrt(Mfluc.Ux2ave) / nu

    Mfluc.lambdaTy_Local = np.sqrt(Mfluc.Uy2ave / Mfluc.dUydy2ave)
    Mfluc.Re_lambdaTy_Local = Mfluc.lambdaTy_Local * np.sqrt(Mfluc.Uy2ave) / nu

    # Mfluc.lambdaT_Local = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    # Mfluc.lambdaTx_Local = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    # Mfluc.lambdaTy_Local = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    # Mfluc.Re_lambdaT_Local = np.zeros((Mfluc.x.shape[0], Mfluc.Ux.shape[1]))
    # Mfluc.Re_lambdaTx_Local = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    # Mfluc.Re_lambdaTy_Local = np.zeros((Mfluc.Ux.shape[0], Mfluc.Ux.shape[1]))
    # for x in range(0, Mfluc.Ux.shape[0] - 1):
    #     for y in range(0, Mfluc.Ux.shape[1] - 1):
    #         if Mfluc.dUidxi2ave[x, y] < cutoff:
    #             Mfluc.lambdaT_Local[x, y] = 0
    #         else:
    #             Mfluc.lambdaT_Local[x, y] = np.sqrt(Mfluc.U2ave[x, y] / Mfluc.dUidxi2ave[x, y])
    #             Mfluc.Re_lambdaT_Local[x, y] = Mfluc.lambdaT_Local[x, y] * np.sqrt(Mfluc.U2ave[x, y]) / nu
    #         if Mfluc.dUxdx2ave[x, y] < cutoff:
    #             Mfluc.lambdaTx_Local[x, y] = 0
    #         else:
    #             Mfluc.lambdaTx_Local[x, y] = np.sqrt(Mfluc.Ux2ave[x, y] / Mfluc.dUxdx2ave[x, y])
    #             Mfluc.Re_lambdaTx_Local[x, y] = Mfluc.lambdaTx_Local[x, y] * np.sqrt(Mfluc.Ux2ave[x, y]) / nu
    #         if Mfluc.dUydy2ave[x, y] < cutoff:
    #             Mfluc.lambdaTy_Local[x, y] = 0
    #         else:
    #             Mfluc.lambdaTy_Local[x, y] = np.sqrt(Mfluc.Uy2ave[x, y] / Mfluc.dUydy2ave[x, y])
    #             Mfluc.Re_lambdaTy_Local[x, y] = Mfluc.lambdaTy_Local[x, y] * np.sqrt(Mfluc.Uy2ave[x, y]) / nu

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



def compute_lambda_Re_lambda(Mfluc, nu=1.004, x0=None, y0=None, x1=None, y1=None, redo=False):
    """
    Compute local lambda and Re_lambda through velocity gradients
    Parameters
    ----------
    Mfluc
    nu

    Returns
    -------

    """
    if x0 is None:
        x0, y0 = 0, 0
        x1, y1 = Mfluc.Ux.shape[1], Mfluc.Ux.shape[0]
    print 'Will average from (%d, %d) to (%d, %d)  (index)' % (x0, y0, x1, y1)

    if not hasattr(Mfluc, 'U2ave') or redo:
        print 'Mfluc does not have attribute Mfluc.U2ave, or redo the computation. Calculate it.'
        Mfluc = compute_u2_duidxi(Mfluc, x0=x0, y0=y0, x1=x1, y1=y1, clean=True)

    Mfluc.lambdaT = np.sqrt(np.nanmean(Mfluc.U2ave[y0:y1, x0:x1])/np.nanmean(Mfluc.dUidxi2ave[y0:y1, x0:x1]))
    Mfluc.lambdaTx = np.sqrt(np.nanmean(Mfluc.Ux2ave[y0:y1, x0:x1])/np.nanmean(Mfluc.dUxdx2ave[y0:y1, x0:x1]))
    Mfluc.lambdaTy = np.sqrt(np.nanmean(Mfluc.Uy2ave[y0:y1, x0:x1])/np.nanmean(Mfluc.dUydy2ave[y0:y1, x0:x1]))

    Mfluc.Re_lambdaT = np.sqrt(np.nanmean(Mfluc.U2ave[y0:y1, x0:x1])) * Mfluc.lambdaT / nu
    Mfluc.Re_lambdaTx = np.sqrt(np.nanmean(Mfluc.Ux2ave[y0:y1, x0:x1])) * Mfluc.lambdaTx / nu
    Mfluc.Re_lambdaTy = np.sqrt(np.nanmean(Mfluc.Uy2ave[y0:y1, x0:x1])) * Mfluc.lambdaTy / nu
    print 'lambdaT, Re_lambdaT, lambdaTx, Re_lambdaTx, lambdaTy, Re_lambdaTy'
    print Mfluc.lambdaT, Mfluc.Re_lambdaT, Mfluc.lambdaTx, Mfluc.Re_lambdaTx, Mfluc.lambdaTy, Mfluc.Re_lambdaTy
    return Mfluc

