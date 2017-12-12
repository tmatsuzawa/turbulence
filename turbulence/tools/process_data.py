"""
Module to clean PIV data stored in Mdata class

"""
import numpy as np
import copy
import numpy.ma as ma


def fill_unphysical(arr1, mask):
    """

    This interpolation is not ideal because this flattens multidimensional array first, and takes a linear interpolation
    for missing values. That is, the interpolated values at the edges of the multidimensional array are nonsense b/c
    actual data does not have a periodic boundary condition.

    """
    arr2T = copy.deepcopy(arr1).T

    f0 = np.flatnonzero(mask)
    f1 = np.flatnonzero(~mask)

    arr1[mask] = np.interp(f0, f1, arr1[~mask])

    f0 = np.flatnonzero(mask.T)
    f1 = np.flatnonzero(~mask.T)
    arr2T[mask.T] = np.interp(f0, f1, arr1.T[~(mask.T)])
    arr2 = arr2T.T

    arr = (arr1 + arr2) * 0.5
    return arr


def get_mask_for_unphysical(U, cutoffU=2000., fill_value = np.nan):
    """

    Parameters
    ----------
    U: array-like
    cutoffU: float
        if |value| > cutoff, this method considers those values unphysical.
    fill_value:


    Returns
    -------
    mask: multidimensional boolean array

    """
    print 'number of invalid values (nan and inf) in the array: ' + str(np.isnan(U).sum() + np.isinf(U).sum())
    print 'number of nan values in U: ' + str(np.isnan(U).sum())
    print 'number of inf values in U: ' + str(np.isinf(U).sum()) + '\n'

    # a=ma.masked_invalid(U)
    # print 'number of masked elements by masked_invalid: '+ str(ma.count_masked(a))

    # Replace all nan and inf values with fill_value.
    # fix_invalid still enforces a mask on elements with originally invalid values
    U_fixed = ma.fix_invalid(U, fill_value=99999)
    n_invalid = ma.count_masked(U_fixed)
    print 'number of masked elements by masked_invalid: ' + str(n_invalid)
    # Update the mask to False (no masking)
    U_fixed.mask = False



    # Mask unreasonable values of U_fixed
    b = ma.masked_greater(U_fixed, cutoffU)
    c = ma.masked_less(U_fixed, -cutoffU)
    n_greater = ma.count_masked(b) - n_invalid
    n_less = ma.count_masked(c)
    print 'number of masked elements greater than cutoff: ' + str(n_greater)
    print 'number of masked elements less than -cutoff: ' + str(n_less)

    mask = ~(~b.mask * ~c.mask)  #this masks all nonsense values of U.
    d = ma.array(U_fixed, mask=mask)
    n_total = ma.count_masked(d)
    # U_filled = ma.filled(d, fill_value)

    #Total number of elements in U
    N = 1
    for i in range(len(U.shape)):
        N *= U.shape[i]

    print 'total number of unphysical values: ' + str(ma.count_masked(d)) + '  (' + str((float(n_total)/N*100)) + '%)\n'


    return mask

def fill_unphysical_with_sth(U, mask, fill_value=np.nan):
    """

    Parameters
    ----------
    U   array-like
    mask   multidimensional boolean array
    fill_value   value that replaces masked values

    Returns
    -------
    U_filled

    """
    U_masked = ma.array(U, mask=mask)
    U_filled = ma.filled(U_masked, fill_value)  # numpy array. This is NOT a masked array.

    return U_filled

def clean_vdata(M, cutoffU=2000, fill_value=np.nan):
    print 'Cleaning M.Ux...'
    mask = get_mask_for_unphysical(M.Ux, cutoffU=cutoffU, fill_value=fill_value)
    Ux_filled_with_nans = fill_unphysical_with_sth(M.Ux, mask, fill_value=fill_value)
    Ux_interpolated = fill_unphysical(Ux_filled_with_nans, mask)
    M.Ux[:]= Ux_interpolated[:]
    print 'Cleaning M.Uy...'
    mask = get_mask_for_unphysical(M.Uy, cutoffU=cutoffU, fill_value=fill_value)
    Uy_filled_with_nans = fill_unphysical_with_sth(M.Uy, mask, fill_value=fill_value)
    Uy_interpolated = fill_unphysical(Uy_filled_with_nans, mask)
    M.Uy[:]= Uy_interpolated[:]
    print 'Cleaning Done.'
    return M