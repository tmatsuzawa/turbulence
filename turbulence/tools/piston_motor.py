"""Module related to piston/ Copley motor
author: takumi"""


def get_exp_params(size='mid'):
    """ Returns geometric information of box
    Parameters
    ----------
    size : str, this could be either 'small' or 'mid'

    Returns
    -------

    """
    if size == 'mid':
        box_width = 325. # mm
        dp = 160. # mm
        do = 25.6 # mm
    if size == 'small':
        box_width = 254. # mm
        dp = 125. # mm
        do = 20. # mm
    return box_width, dp, do

def compute_LD(span=1, dp=1, do=1, norfice=8,  size=None):
    """ Returns a L/D value for given span, do, dp, and nuorfice. One can also use 'size' to get preset values.

    Parameters
    ----------
    span : stroke length in mm
    do : orfice diameter in mm
    dp : piston diameter in mm
    norfice : number of orfices in a box
    size: option to get preset geometric values of box

    Returns
    -------

    """
    if size is not None:
        box_width, dp, do = get_exp_params(size)
    ld = (dp / do)**2 / do * span / norfice
    return ld


