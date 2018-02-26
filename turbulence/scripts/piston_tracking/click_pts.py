import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import argparse
import glob
import numpy as np
import lepm.lattice_elasticity as le
import cPickle as pickle

'''Click on a point and get its coordinate
'''


class ImagePoint:
    def __init__(self, img=None, fn=None):
        if img is None and fn is None:
            raise RuntimeError('Supply either img or fn kwarg')
        if img is not None:
            self.img = img
        elif fn is not None:
            self.img = mpimg.imread(fn)
        self.point = ()
        self.fig = None

    def getCoord(self, xyprev=None, buff=0, figsize=(20, 15)):
        self.fig = plt.figure(figsize=figsize)
        ax = self.fig.add_subplot(111)
        plt.imshow(self.img, cmap=cm.gray)
        cid = self.fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        if xyprev is not None and buff > 0:
            print 'zooming in...'
            plt.xlim(xyprev[0] - buff, xyprev[0] + buff)
            plt.ylim(xyprev[1] - buff, xyprev[1] + buff)
            plt.plot([xyprev[0]], [xyprev[1]], 'ro')
        plt.show()
        print self.point
        return self.point

    def __onclick__(self, click):
        self.point = (click.xdata, click.ydata)
        return self.point

    def closefig(self):
        # raw_input("Press [anything] to continue.") # wait for input from the user
        # close the figure to show the next one.
        print self.point
        plt.close(self.fig)
