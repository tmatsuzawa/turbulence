#! python
from scipy import optimize
import numpy as np

'''Fit some data to a circle using scipy'''


def fit_pts_to_circle(xx, yy):
    """Given some data (xx,yy) that lives on/near a circle, compute the center and radius of the circle.

    Parameters
    ----------
    xx :
    yy :

    Returns
    -------
    xc_2b, yc_2b, R_2b, residu_2b
    """
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    def Df_2b(c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        df2b_dc = np.empty((len(c), x.size))

        Ri = calc_R(xc, yc)
        df2b_dc[0] = (xc - xx) / Ri  # dR/dxc
        df2b_dc[1] = (yc - yy) / Ri  # dR/dyc
        df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

        return df2b_dc

    # Estimate the centers as the mean of the data
    # coordinates of the barycenter
    x_m = np.mean(xx)
    y_m = np.mean(yy)

    center_estimate = x_m, y_m
    center_2b, ier = optimize.leastsq(f_2, center_estimate, Dfun=Df_2b, col_deriv=True)

    xc_2b, yc_2b = center_2b
    Ri_2b = calc_R(*center_2b)
    R_2b = Ri_2b.mean()
    residu_2b = np.sum((Ri_2b - R_2b)**2)
    return xc_2b, yc_2b, R_2b, residu_2b

