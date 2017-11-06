"""
Module to compute various quantities that characterize turbulence
author: takumi
date: 11/02/17
"""

import numpy as np
import turbulence.display.graphes as graphes
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def func1(x, a, b,c):
    return a*np.exp(-b*x)+c


def autocorrelation(M, plot=True, fit=False):
    Nlags = int(np.floor((M.Ux.shape[2] - 1) / 2.))

    # Compute the autocorrelation function at each position, rho(x,y)
    lag = np.zeros((M.Uy.shape[0], M.Uy.shape[1], Nlags * 2 + 1))
    rho = np.zeros((M.Uy.shape[0], M.Uy.shape[1], Nlags * 2 + 1))
    Dt = np.mean(np.diff(M.t))
    for x in range(0, M.Uy.shape[0]):
        for y in range(0, M.Uy.shape[1]):
            A = plt.acorr(M.Uy[x, y, ...], maxlags=Nlags)
            a = list(A[0])
            b = list(A[1])
            for i in range(len(a)):
                a[i] = float(a[i]) * Dt  # convert lag[frame] into lag[sec]
                lag[x][y][i] = a[i]
                rho[x][y][i] = b[i]
        print x
    rho_mean = np.mean(rho, axis=(0, 1)) #spatial average of autocorrelation function

    # Convert 3d numpy array to 1d array for plotting
    lag = list(A[0])
    print lag

    # Convert lag[frame] into lag[]
    for t in range(len(lag)):
        lag[t] = float(lag[t]) * Dt


    # Plot the spatially averaged autocorrelation function
    if plot:
            fig1 = plt.figure()
            ax = fig1.add_subplot(1, 1, 1)
            plt.axis([-3, 3, 0, 1])
            plt.plot(lag, rho_mean)
            plt.xlabel('$\\tau=t-t\'$ [s]')
            plt.ylabel('$\\rho(\\tau)$')
    # Fit the spatially averaged autocorrelation function with exponential
    if fit:
            a = lag[int(Nlags):]
            X = [np.mean(a[i]) for i in range(0, int(Nlags / 2))]
            b = rho_mean[int(Nlags):]
            Y = [np.mean(b[i]) for i in range(0, int(Nlags / 2))]
            popt, pcov = curve_fit(func1, X, Y, bounds=([0, 0, 0], [2, 100, 0.5]))

            x = np.arange(0., 1., 0.0001)
            y = func1(x, *popt)
            plt.plot(x, y, 'r--', label='fit')
            fit_param = list()
            for item in popt:
                fit_param.append(str(round(item, 3)))
            fit_eq = '$y=a*exp(-b*x)+c$:      $a$=' + fit_param[0] + ', $b$=' + fit_param[1] + ', $c$=' + fit_param[2]
            ax.text(-2.1, 0.8, fit_eq, fontsize=10)
            tau_half = np.log(2) / float(fit_param[1])
            print 'tau1/2 = ' + str(tau_half) + ' [s]'
            print 'fit parameters: a*exp(-b*x)+c:'
            print popt

    return lag, rho_mean

def compute_intTimeScale(M,lag,rho_mean,plot=True):
    Dt = np.diff(M.t)
    x0 = (len(lag) - 1) / 2
    lag_plus = [lag[x] for x in range((len(lag) - 1) / 2, len(lag) - 1)]
    Tau = np.zeros(len(lag_plus))

    xlist = range((len(lag) - 1) / 2, len(lag) - 1)
    for x in range((len(lag) - 1) / 2, len(lag) - 1):
        if x == 0:
            Tau[x - x0] = Dt[x] * rho_mean[x]
        else:
            Tau[x - x0] = Tau[x - x0 - 1] + Dt[x] * rho_mean[x]
    # Plot the integrated value of spatially averaged autocorrelation function between [0, tau]
    if plot:
        plt.figure()
        plt.plot(lag_plus, Tau)
        plt.xlabel('$\\tau=t-t\'$ [s]')
        plt.ylabel('integral')

    IntScaleTime = np.max(Tau)

    print 'The total integration value is ' + str(IntScaleTime)
    return IntScaleTime
