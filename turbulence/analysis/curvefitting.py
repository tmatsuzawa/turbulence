from scipy.optimize import curve_fit

'''Shell of a module which echos scipy commands
'''


def fit_data_curvefit(function, xdata, ydata, yerr=None, **kwargs):
    """
    Use non-linear least squares to fit a function to data.

    Assumes ``ydata = f(xdata, *params) + eps``

    Parameters
    ----------
    function : callable
        The model function, f(x, ...).  It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
    xdata : An M-length sequence or an (k,M)-shaped array for functions with k predictors
        The independent variable where the data is measured.
    ydata : M-length sequence
        The dependent data --- nominally f(xdata, ...)
    yerr : M-length sequence
        the uncertainty on the ydata, passed as sigma to curve_fit()
    p0 : None, scalar, or N-length sequence, optional
        Initial guess for the parameters.  If None, then the initial
        values will all be 1 (if the number of parameters for the function
        can be determined using introspection, otherwise a ValueError
        is raised).
    sigma : None or M-length sequence or MxM array, optional
        Determines the uncertainty in `ydata`. If we define residuals as
        ``r = ydata - f(xdata, *popt)``, then the interpretation of `sigma`
        depends on its number of dimensions:

            - A 1-d `sigma` should contain values of standard deviations of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = sum((r / sigma) ** 2)``.

            - A 2-d `sigma` should contain the covariance matrix of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = r.T @ inv(sigma) @ r``.

              .. versionadded:: 0.19

        None (default) is equivalent of 1-d `sigma` filled with ones.
    absolute_sigma : bool, optional
        If True, `sigma` is used in an absolute sense and the estimated parameter
        covariance `pcov` reflects these absolute values.

        If False, only the relative magnitudes of the `sigma` values matter.
        The returned parameter covariance matrix `pcov` is based on scaling
        `sigma` by a constant factor. This constant is set by demanding that the
        reduced `chisq` for the optimal parameters `popt` when using the
        *scaled* `sigma` equals unity. In other words, `sigma` is scaled to
        match the sample variance of the residuals after the fit.
        Mathematically,
        ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``
    check_finite : bool, optional
        If True, check that the input arrays do not contain nans of infs,
        and raise a ValueError if they do. Setting this parameter to
        False may silently produce nonsensical results if the input arrays
        do contain nans. Default is True.
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on independent variables. Defaults to no bounds.
        Each element of the tuple must be either an array with the length equal
        to the number of parameters, or a scalar (in which case the bound is
        taken to be the same for all parameters.) Use ``np.inf`` with an
        appropriate sign to disable bounds on all or some parameters.

        .. versionadded:: 0.17
    method : {'lm', 'trf', 'dogbox'}, optional
        Method to use for optimization.  See `least_squares` for more details.
        Default is 'lm' for unconstrained problems and 'trf' if `bounds` are
        provided. The method 'lm' won't work when the number of observations
        is less than the number of variables, use 'trf' or 'dogbox' in this
        case.

        .. versionadded:: 0.17
    jac : callable, string or None, optional
        Function with signature ``jac(x, ...)`` which computes the Jacobian
        matrix of the model function with respect to parameters as a dense
        array_like structure. It will be scaled according to provided `sigma`.
        If None (default), the Jacobian will be estimated numerically.
        String keywords for 'trf' and 'dogbox' methods can be used to select
        a finite difference scheme, see `least_squares`.

        .. versionadded:: 0.18
    kwargs
        Keyword arguments passed to `leastsq` for ``method='lm'`` or
        `least_squares` otherwise.

    Returns
    -------
    popt : array
        Optimal values for the parameters so that the sum of the squared
        residuals of ``f(xdata, *popt) - ydata`` is minimized
    pcov : 2d array
        The estimated covariance of popt. The diagonals provide the variance
        of the parameter estimate. To compute one standard deviation errors
        on the parameters use ``perr = np.sqrt(np.diag(pcov))``.

        How the `sigma` parameter affects the estimated covariance
        depends on `absolute_sigma` argument, as described above.

        If the Jacobian matrix at the solution doesn't have a full rank, then
        'lm' method returns a matrix filled with ``np.inf``, on the other hand
        'trf'  and 'dogbox' methods use Moore-Penrose pseudoinverse to compute
        the covariance matrix.

    Raises
    ------
    ValueError
        if either `ydata` or `xdata` contain NaNs, or if incompatible options
        are used.

    RuntimeError
        if the least-squares minimization fails.

    OptimizeWarning
        if covariance of the parameters can not be estimated.

    See Also
    --------
    least_squares : Minimize the sum of squares of nonlinear functions.
    scipy.stats.linregress : Calculate a linear least squares regression for
                             two sets of measurements.

    Notes
    -----
    With ``method='lm'``, the algorithm uses the Levenberg-Marquardt algorithm
    through `leastsq`. Note that this algorithm can only deal with
    unconstrained problems.

    Box constraints can be handled by methods 'trf' and 'dogbox'. Refer to
    the docstring of `least_squares` for more information.

    Examples
    --------
    $ import numpy as np
    $ import matplotlib.pyplot as plt
    $ from scipy.optimize import curve_fit

    $ def func(x, a, b, c):
    ...     return a * np.exp(-b * x) + c

    define the data to be fit with some noise

    $ xdata = np.linspace(0, 4, 50)
    $ y = func(xdata, 2.5, 1.3, 0.5)
    $ y_noise = 0.2 * np.random.normal(size=xdata.size)
    $ ydata = y + y_noise
    $ plt.plot(xdata, ydata, 'b-', label='data')

    Fit for the parameters a, b, c of the function `func`

    $ popt, pcov = curve_fit(func, xdata, ydata)
    $ plt.plot(xdata, func(xdata, *popt), 'r-', label='fit')

    Constrain the optimization to the region of ``0 < a < 3``, ``0 < b < 2``
    and ``0 < c < 1``:

    $ popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 2., 1.]))
    $ plt.plot(xdata, func(xdata, *popt), 'g--', label='fit-with-bounds')

    $ plt.xlabel('x')
    $ plt.ylabel('y')
    $ plt.legend()
    $ plt.show()

    """
    popt, pcov = curve_fit(function, xdata, ydata, sigma=yerr, **kwargs)
    return popt, pcov


def linear_fit(xdata, ydata, yerr=None):
    """

    Parameters
    ----------
    xdata:
    ydata:
    yerr : float, n x 1 float array, or None
        The error associated with each y data point

    Returns
    -------
    popt : best fit for [slope, intercept]
    pcov : covariance for fit of [slope, intercept]
    """
    def linear(x, a, b):
        return a * x + b

    popt, pcov = fit_data_curvefit(linear, xdata, ydata, yerr)
    return popt, pcov


def fit_log_with_cutoff(xx, yy, intensities, intensity_err=None, rlower=0.):
    """Fit

    Parameters
    ----------
    xx : N x N float array
        x positions of the image
    yy : N x N float array
        y positions of the image
    intensities : N x N float array
        the intensities to be fit

    Returns
    -------
    popt :
        slope, intercept for log-log fit
    pcov :
        uncertainties for slope, intercept for log-log fit
    """
    # get radial position from xy
    rr = (xx ** 2 + yy ** 2).ravel()
    ii = intensities.ravel()[rr > rlower]
    rr = rr[rr > rlower]
    # fit it on log-log, so take log of each and institute cutoff
    logr = np.log(rr)
    logi = np.log(ii)
    if intensity_err is None:
        yerr = None
    else:
        yerr = intensity_err.ravel()[rr > rlower]
    popt, pcov = linear_fit(logr, logi, yerr=yerr)
    return popt, pcov


if __name__ == '__main__':
    """Example usage"""
    import numpy as np
    import matplotlib.pyplot as plt

    # make some test data
    xx = np.arange(100.) - 50.
    yy = np.arange(100.) - 50.
    xgrid, ygrid = np.meshgrid(xx, yy)
    sigma, cutoff = 20., 0.5
    blob = (xgrid ** 2 + ygrid **2) ** (-4) * 1e6
    # chop off intensity to simulate having a flat feature in the intensity near the center
    blob[blob > cutoff] = cutoff

    # fit it
    cutoff = 100
    rr = (xgrid ** 2 + ygrid ** 2).ravel()
    ii = blob.ravel()
    popt, pcov, = fit_log_with_cutoff(xgrid, ygrid, blob, intensity_err=None, rlower=cutoff)

    # plot results
    plt.loglog(rr, ii, '.')
    # ln(y) = A ln(x) + B
    # y = e^B x^A
    a, b = popt[0], popt[1]
    xtmp = np.linspace(cutoff, np.max(rr), 100)
    ytmp = np.exp(b) * xtmp ** a
    plt.loglog(xtmp, ytmp, '--')
    plt.plot([cutoff, cutoff], [np.min(ii), np.max(ii)], 'k--')
    plt.title('Fit to powerlaw decay with cutoff')
    plt.xlabel('distance')
    plt.ylabel('intensity')
    plt.show()