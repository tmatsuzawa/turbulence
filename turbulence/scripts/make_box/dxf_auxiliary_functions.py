import numpy as np
import matplotlib.pyplot as plt
import sys


'''Auxiliary functions for making the polygons of the box'''


def polygon2sawgon(polygon, wtooth, wspace, htooth=6.35, lineseg_inds=None, outward=None, invert=None,
                   offsets=None, check=False):
    """Given a polygon, decorate its border with a sawtooth pattern.

    Parameters
    ----------
    polygon : N x 2 float array
        the vertices of the polygons to decorate with sawtooth boundary
    wtooth : float
        the width of each tooth
    wspace : float
        the width of each space between teeth
    htooth : float
        the extent of each tooth protrusion
    lineseg_inds : list of lists of ints or None
        the pairs of indices of polygon to take as endpoints of linesegments that are to be decorated
    outward : bool or list of bools or None

    offsets : length-M list or array or None


    Returns
    -------

    """
    # form pairs if lineseg_inds is None
    if lineseg_inds is None:
        lineseg_inds = [[ii, (ii + 1) % len(polygon)] for ii in range(len(polygon))]
        print 'lineseg_inds = ', lineseg_inds
    if outward is None:
        outward = np.ones(len(lineseg_inds), dtype=bool)
    elif isinstance(outward, bool):
        outward = int(outward) * np.ones(len(lineseg_inds), dtype=int)
    if offsets is None:
        offsets = [0. for seg in lineseg_inds]
    if invert is None:
        invert = np.zeros(len(lineseg_inds), dtype=bool)
    elif isinstance(invert, bool):
        if invert is True:
            invert = np.ones(len(lineseg_inds), dtype=bool)
        else:
            invert = np.zeros(len(lineseg_inds), dtype=bool)
    # Otherwise invert is a list or array matching lineseg_inds

    # if invert:
    #     # flip the order of the polygon lineseg indices
    #     polygon = polygon[::-1]
    #     kk = 0
    #     print 'switching order of lineseg_inds'
    #     for pair in lineseg_inds:
    #         lineseg_inds[kk] = [len(polygon) - pair[0] - 1, len(polygon) - pair[1] - 1]
    #         kk += 1

    print 'lineseg_inds = ', lineseg_inds
    print 'len(polygon) = ', len(polygon)
    # sys.exit()
    paths = []
    kk = 0
    lsind = 0
    done = False
    while not done:
        print 'lsind = ', lsind
        # print 'lineseg_inds = ', lineseg_inds
        if lsind >= len(lineseg_inds):
            pair = [0, 0]
        else:
            pair = lineseg_inds[lsind]
            print 'pair = ', pair
        # comparison is the current index to check if pair[0] is equal to
        # if invert:
        #     comparison = len(polygon) - kk - 1
        # else:
        comparison = kk

        if pair[0] == comparison:
            lineseg = polygon[pair]
            print 'pair = ', pair
            print 'offsets[lsind] = ', offsets[lsind]
            path = square_wave_path(lineseg, wtooth, wspace, htooth=htooth, outward=outward[lsind],
                                    invert=invert[lsind], offset=offsets[lsind], check=check)
            paths.append(path)
            if pair[1] == 0:
                done = True
            lsind += 1
        else:
            paths.append(polygon[kk:kk + 2])
            # print 'paths = ', paths
            # print 'len(polygon) = ', len(polygon)
            # print 'kk = ', kk
            if len(polygon) == kk + 1:
                done = True
        kk += 1

    if check:
        for path in paths:
            plt.plot(path[:, 0], path[:, 1], '.-')

        plt.title('offset = ' + str(np.max(np.abs(offsets))))
        plt.axis('equal')
        plt.show()

    return np.vstack((tuple(paths)))


def square_wave_path(lineseg, wtooth, wspace, htooth=6.35, outward=True, invert=False, offset=0.,
                     check=False, eps=1e-7):
    """Given a lineseg, return a square wave path from endpoint0 to endpoint1.
    Midpoint of the lineseg is protruded (outward=True) or not (outward=False)

    Parameters
    ----------
    lineseg : 4 x 1 or 2 x 2 float array
        the endpoints of the linesegment
    wtooth : float
        the width of each tooth, in mm
    wspace : float
        the width of the gap between teeth, in mm
    htooth : float
        the height of the tooth, in mm
    outward : bool
        this bool determines whether the center of the lineseg is protruding or not
    invert : bool
        Whether the protrusions become intrusions
    offset : float
        Offset along the direction from endpt0 to endpt1 (or reversed if invert is True) for the square wave

    Returns
    -------
    path : n x 2 float array
        the path in xy coordinates of the square wave path from endpoint0 to endpoint1
    """
    # Allow lineseg argument to be 4 x 1 or 2 x 2
    if len(np.shape(lineseg)) == 1 or np.shape(lineseg)[0] == 1:
        # lineseg is 4 x 1
        end0 = np.array([lineseg[0], lineseg[1]])
        end1 = np.array([lineseg[2], lineseg[3]])
    else:
        end0 = np.array([lineseg[0, 0], lineseg[0, 1]])
        end1 = np.array([lineseg[1, 0], lineseg[1, 1]])

    # if invert:
    #     end1, end0 = end0, end1
    invert = np.sign(0.5 - int(invert))

    # print 'outward = ', outward
    # print 'outward = ', type(outward)
    outward = int(outward)
    if outward > 0 and invert < 0:
        # since we merely switch the order of which we add, swap the names of tooth and space
        wtooth, wspace = wspace, wtooth
    # if invert:
    #     wtooth, wspace = wspace, wtooth

    dx = end1[0] - end0[0]
    dy = end1[1] - end0[1]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    theta = np.arctan2(dy, dx)
    # create path, cut up into segments of wtooth + wspace
    pr, pl = [], []
    if wtooth > dist:
        pr = [[dist * 0.5, 0]]
        pl = [[-dist * 0.5, 0]]
    else:
        if abs(offset) < eps:
            xx, ss = 0, 0
            while ss < dist * 0.5:
                ss = min(dist * 0.5, xx + wtooth * 0.5)
                # make right segments
                pr.append([ss, htooth * outward * invert])
                if ss < dist * 0.5 - eps:
                    pr.append([ss, htooth * ((outward + 1) % 2) * invert])

                # make left segments
                pl.append([-ss, htooth * outward * invert])
                if ss < dist * 0.5 - eps:
                    pl.append([-ss, htooth * ((outward + 1) % 2) * invert])

                if ss < dist * 0.5 - eps:
                    ss = min(dist * 0.5, xx + wtooth * 0.5 + wspace)
                    # make right space segment
                    pr.append([ss, htooth * ((outward + 1) % 2) * invert])
                    pr.append([ss, htooth * outward * invert])
                    # make left space segment
                    pl.append([-ss, htooth * ((outward + 1) % 2) * invert])
                    pl.append([-ss, htooth * outward * invert])

                xx += wtooth + wspace
        else:
            ssr, xx = 0., 0.
            while ssr < dist * 0.5:
                ssr = min(dist * 0.5, xx + wtooth * 0.5 + offset)
                print 'ssr = ', ssr
                # make right segments
                pr.append([ssr, htooth * outward * invert])
                if ssr < dist * 0.5 - eps:
                    pr.append([ssr, htooth * ((outward + 1) % 2) * invert])

                if ssr < dist * 0.5 - eps:
                    ssr = min(dist * 0.5, xx + wtooth * 0.5 + wspace + offset)
                    # make right space segment
                    pr.append([ssr, htooth * ((outward + 1) % 2) * invert])
                    pr.append([ssr, htooth * outward * invert])

                xx += wtooth + wspace

            if check:
                print 'pr = ', pr

            ssl, xx = 0., 0.
            while ssl < dist * 0.5:
                ssl = min(dist * 0.5, xx + wtooth * 0.5 - offset)
                # make left segments
                pl.append([-ssl, htooth * outward * invert])
                if ssl < dist * 0.5 - eps:
                    pl.append([-ssl, htooth * ((outward + 1) % 2) * invert])

                # make left space segment
                if ssl < dist * 0.5 - eps:
                    ssl = min(dist * 0.5, xx + wtooth * 0.5 + wspace - offset)
                    pl.append([-ssl, htooth * ((outward + 1) % 2) * invert])
                    pl.append([-ssl, htooth * outward * invert])

                xx += wtooth + wspace

            if check:
                print 'pl = ', pl

    if pr[-1] != [dist * 0.5, 0.]:
        pr.append([dist * 0.5, 0.])
    if pl[-1] != [-dist * 0.5, 0.]:
        pl.append([-dist * 0.5, 0.])

    # print 'pr = ', pr
    # if invert:
    #     pr = np.array(pr)[::-1]
    #     pl = np.array(pl)
    # else:
    pr = np.array(pr)
    pl = np.array(pl)[::-1]

    path = np.vstack((pl, pr))
    print 'path = ', path
    path = rotate_vectors_2d(path + np.array([dist * 0.5, 0]), theta)
    path += end0

    return path


def rotate_vectors_2d(xy, theta):
    """Given an array of vectors, rotate them actively counterclockwise in the xy plane by theta.

    Parameters
    ----------
    xy : NP x 2 array
        Each row is a 2D x,y vector to rotate counterclockwise
    theta : float
        Rotation angle

    Returns
    ----------
    xyrot : NP x 2 array
        Each row is the rotated row vector of XY

    """
    rr = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    xyrot = np.dot(rr, xy.transpose()).transpose()
    return xyrot


def save_dict(pdict, filename, header, keyfmt='auto', valfmt='auto', padding_var=7):
    """Writes dictionary to txt file where each line reads 'key    : value'.

    Parameters
    ----------
    Pdict : dict
        dictionary of key, value pairs to write as txt file
    header : string
        header for the text file specifying content of the file
    keyfmt : string
        string formatting for keys, by default this is 'auto'. If not 'auto', then all keys are formatted identically.
    valfmt : string
        string formatting for value, by default this is 'auto'. If not 'auto', then all values are formatted identically.

    Returns
    ----------
    """
    with open(filename, 'w') as myfile:
        if '#' in header:
            myfile.write(header + '\n')
        else:
            myfile.write('# ' + header + '\n')

    # if keyfmt == 'auto' and valfmt == 'auto':
    for key in pdict:
        with open(filename, 'a') as myfile:
            # print 'Writing param ', str(key)
            # print ' with value ', str(pdict[key])
            # print ' This param is of type ', type(pdict[key])
            if isinstance(pdict[key], str):
                myfile.write('{{0: <{}}}'.format(padding_var).format(str(key)) + '= ' + pdict[key] + '\n')
            elif isinstance(pdict[key], np.ndarray):
                myfile.write('{{0: <{}}}'.format(padding_var).format(str(key)) +
                             '= ' + ", ".join(np.array_str(pdict[key], precision=18).split()).replace('[,', '[') + '\n')
            elif isinstance(pdict[key], list):
                myfile.write('{{0: <{}}}'.format(padding_var).format(str(key)) + '= ' + str(pdict[key]) + '\n')
            elif isinstance(pdict[key], tuple):
                myfile.write('{{0: <{}}}'.format(padding_var).format(str(key)) + '= ' + str(pdict[key]) + '\n')
            elif pdict[key] is None:
                # Don't write key to file if val is None
                pass
                # myfile.write('{{0: <{}}}'.format(padding_var).format(str(key)) + '= none' + '\n')
            else:
                # print 'dio.save_dict(): ', key, ' = ', pdict[key]
                # print 'isstr --> ', isinstance(pdict[key], str)
                myfile.write('{{0: <{}}}'.format(padding_var).format(str(key)) +
                             '= ' + '{0:0.18e}'.format(pdict[key]) + '\n')
