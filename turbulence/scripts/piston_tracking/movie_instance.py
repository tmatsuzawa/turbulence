import cine
import cv2
import numpy as np
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
import math
import PIL.Image as Image
import PIL.ImageOps as ImageOps
import matplotlib.cm as cm
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import matplotlib.image as mpimg
import tracking_helper_functions as thf
import lepm.data_handling as dh
import copy
from lepm.build.roipoly import RoiPoly
import lepm.data_handling as dh
import lepm.plotting.plotting as leplt
from click_pts import ImagePoint
from scipy.ndimage import filters


'''
'''


class GyroMovie:
    def __init__(self, input_file, frame_rate=40.):
        # first determine the file type (cine, tiff, png, etc)
        self.file_type = input_file.split('.')[-1]

        if self.file_type == 'cine':
            self.cine = True
            self.data = cine.Cine(input_file)
            self.num_frames = len(self.data)
        else:
            self.cine = False
            file_names = thf.find_files_by_extension(input_file, '.png', tot=True)
            data = []
            for i in xrange(20):
                file = file_names[i]
                data_single = mpimg.imread(file)
                data_single = self.rgb2gray(data_single)
                data_single = data_single[:, 100:1400]
                data.append(data_single)
                print np.shape(data_single)
                print file
            self.data = data
            self.num_frames = len(self.data)

        # print 'data = ', self.data
        # print 'type(data) = ', type(self.data)
        # print 'data[0] = ', self.data[0]
        # print 'type(data[0]) = ', type(self.data[0])
        # sys.exit()

        self._mean_value = 0
        self.min_radius = 17
        self.max_radius = 22
        self.min_dist = 20
        self._min_value = 0.05
        self._max_value = 0.7
        self._pix = 6
        self._centroid_clipmin = 0.0
        self._centroid_clipmax = 1.0

        self.current_frame = []
        self.previous_frame = None
        self.average_frame = None
        self.variance_frame = None
        self.normfactor_frame = None
        self.maxdiff_frame = None
        self.roi = None
        self.frame_current_points = []
        self.reference_points = None
        self.reference_mask = None
        self.ref_masks = []
        self.circles = []
        self.current_time = 0
        self.frame_rate = frame_rate

        self._adjust_min_max_val()
        self._set_dummy_frame()

    def _adjust_min_max_val(self):
        max = np.max(self.data[0].astype('float').flatten())
        self._min_value = self._min_value * max
        self._max_value = self._max_value * max

    def _set_dummy_frame(self):
        t2 = np.ones((2 * self._pix, 2 * self._pix), dtype='f')
        self.dummy = np.array(ndimage.measurements.center_of_mass(t2.astype(float)))

    def set_average_frame(self, first_ind=0, last_ind=None):
        """

        Parameters
        ----------
        first_ind : int
            The first frame number to average over
        last_ind : int
            The last frame number to average over

        Returns
        -------
        self.average_frame : nxm float array
            The average of the data from first_ind to last_ind
        """
        if last_ind is None:
            last_ind = self.num_frames - 1

        for ii in range(first_ind, last_ind + 1):
            if ii == 0:
                avgframe = self.data[ii].astype('float')
            else:
                avgframe += self.data[ii].astype('float')

        num_avgframes = last_ind - first_ind + 1
        self.average_frame = avgframe / num_avgframes
        return self.average_frame

    def set_normfactor_frame(self, normbg_size_frac, first_ind=0, last_ind=None):
        """

        Parameters
        ----------
        first_ind : int
            The first frame number to average over
        last_ind : int
            The last frame number to average over

        Returns
        -------
        self.average_frame : nxm float array
            The average of the data from first_ind to last_ind
        """
        # first make the average over all frames in range
        if last_ind is None:
            last_ind = self.num_frames - 1

        for ii in range(first_ind, last_ind + 1):
            if ii == 0:
                avgframe = self.data[ii].astype('float')
            else:
                avgframe += self.data[ii].astype('float')

        num_avgframes = last_ind - first_ind + 1
        self.average_frame = avgframe / num_avgframes

        # Now set each pixel to its average over a window of size (np.shape(frame) * normbg_size_frac)
        size = np.shape(self.current_frame)[0]
        avgframe = filters.uniform_filter(self.average_frame, size=size, mode='reflect')
        self.normfactor_frame = np.mean(avgframe.ravel()) / avgframe

        # Check it
        # plt.clf()
        # sm = plt.imshow(self.normfactor_frame)
        # plt.colorbar(sm)
        # plt.show()
        # sys.exit()

        return self.normfactor_frame

    def set_variance_frame(self, first_ind=0, last_ind=None):
        """Compute a frame that where each pixel is the variance of that pixel's brightness over the course of frames
        first_ind to last_ind

        Parameters
        ----------
        first_ind : int
            The first frame number to consider
        last_ind : int
            The last frame number to consider

        Returns
        -------
        self.average_frame : nxm float array
            The average of the data from first_ind to last_ind
        """
        if last_ind is None:
            last_ind = self.num_frames - 1

        self.variance_frame = np.var(self.data[first_ind:last_ind + 1], axis=0)
        print 'movie_instance: np.shape(self.variance_frame) = ', np.shape(self.variance_frame)
        # sys.exit()
        return self.variance_frame

    def set_maxdiff_frame(self, first_ind=0, last_ind=None):
        """Compute a frame that where each pixel is the maximum difference between that pixel's brightness in
        consecutive frames over the course of frames first_ind to last_ind.

        Parameters
        ----------
        first_ind : int
            The first frame number to consider
        last_ind : int
            The last frame number to consider

        Returns
        -------
        self.average_frame : nxm float array
            The average of the data from first_ind to last_ind
        """
        if last_ind is None:
            last_ind = self.num_frames - 1

        self.maxdiff_frame = np.median(np.abs(np.diff(self.data[first_ind:last_ind + 1], axis=0)), axis=0)
        print 'movie_instance: np.shape(self.maxdiff_frame) = ', np.shape(self.maxdiff_frame)
        return self.maxdiff_frame

    def set_min_max_val(self, min_value, max_value):
        self._min_value = min_value
        self._max_value = max_value
        self._adjust_min_max_val()

    def set_tracking_size(self, pix):
        """Update pix attribute"""
        self._pix = pix
        self._set_dummy_frame()

    def set_centroid_clip(self, clipmin, clipmax):
        self._centroid_clipmin = clipmin
        self._centroid_clipmax = clipmax

    def update_reference_pts(self):
        self.reference_points = self.frame_current_points
        return self.reference_points

    def set_reference_mask(self, mask=None, mask_method=None, thresh=15., check=False):
        """Define a mask which is True for all pixels further than thresh distance away from reference_points.
        This would be used fro blacking out the image away from each gyro.

        Parameters
        ----------
        mask: n x m bool or int array or None
            the mask to apply to each image. If None, creates mask using specified mask_method
        mask_method : str or None
            If None or 'cf_coords', uses current particle coordinates with radius thresh to set the mask
            Otherwise, lets the user click to make the mask by hand

        Returns
        -------

        """
        if mask is not None:
            self.reference_mask = mask
        else:
            xgrid, ygrid = np.meshgrid(np.arange(np.shape(self.data[0])[0]), np.arange(np.shape(self.data[0])[1]),
                                       sparse=True)
            if mask_method in [None, 'cf_coords']:
                # Define where we are within thresh of reference_points
                for ii in range(len(self.reference_points)):
                    if len(np.shape(self.reference_points)) == 1:
                        dists_ii = (ygrid - self.reference_points[0]) ** 2 + (xgrid - self.reference_points[1]) ** 2
                    else:
                        dists_ii = (ygrid - self.reference_points[ii, 0])**2 + (xgrid - self.reference_points[ii, 1])**2

                    self.ref_masks.append((dists_ii < thresh ** 2).T)
                    if ii == 0:
                        self.reference_mask = copy.deepcopy(self.ref_masks[-1])
                    else:
                        self.reference_mask = np.logical_or(self.ref_masks[-1], self.reference_mask)
            else:
                # set the mask by clicking polygons
                # click on the points to define the unmasked regions
                maskrois = np.zeros_like(self.current_frame, dtype=bool)
                print 'movie_instance: click to define ' + str(int(np.size(self.reference_points) * 0.5)) + ' regions'
                print '(self.reference_points = ', self.reference_points, ')'
                if self.average_frame is None:
                    self.set_average_frame()

                for ii in range(int(np.size(self.reference_points) * 0.5)):
                    plt.close('all')
                    plt.imshow(self.average_frame)
                    ax = plt.gca()
                    polygon = RoiPoly(ax=ax, roicolor='r')
                    maskrois = np.logical_or(maskrois, polygon.get_mask(self.average_frame))
                    # maskrois.append(np.dstack((polygon.allxpoints, polygon.allypoints))[0])

                plt.close('all')
                # Use maskrois to define the mask
                self.reference_mask = maskrois

        # if check:
        #     im = plt.imshow(dists_ii)
        #     plt.colorbar()
        #     plt.show()
        #     plt.clf()
        #     im = plt.imshow(self.reference_mask)
        #     plt.plot(self.reference_points[:, 0], self.reference_points[:, 1], 'r.')
        #     for jj in range(len(self.reference_points)):
        #         plt.text(self.reference_points[jj, 0], self.reference_points[jj, 1], str(jj))
        #     plt.colorbar()
        #     plt.show()
        #     plt.clf()
        if check:
            fig = plt.figure()
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax1.imshow(self.current_frame)
            ax1.plot(self.reference_points[:, 0], self.reference_points[:, 1], 'r.')
            ax2.imshow(self.reference_mask)
            ax2.plot(self.reference_points[:, 0], self.reference_points[:, 1], 'r.')
            plt.show()
            print self.reference_mask
            # sys.exit()

        return self.reference_mask

    def mask_current_frame(self):
        """"""
        # plt.imshow(self.current_frame)
        # plt.show()
        # plt.close('all')
        # print 'self.reference_mask = ', self.reference_mask
        # print 'np.where(self.reference_mask) = ', np.where(self.reference_mask)
        if self.reference_mask is None:
            print 'movie_instance.mask_current_frame(): no reference mask available, skipping masking of current frame...'
        else:
            masked_frame = np.zeros_like(self.current_frame)
            masked_frame[np.where(self.reference_mask)] = self.current_frame[np.where(self.reference_mask)]
            self.current_frame = masked_frame
        # plt.imshow(self.current_frame)
        # plt.show()

    def extract_frame_data(self, frame_num):
        """Set current frame and time"""
        self.current_frame = self.data[frame_num].astype('float')
        self.get_time(frame_num)

    def get_time(self, frame_num):
        # if self.cine:
        #     self.current_time = self.data.get_time(frame_num)
        #
        # else:
        # print('...frame rate set to %02d...' % self.frame_rate)
        self.current_time = 1. / self.frame_rate * frame_num

    def adjust_frame(self):
        """Adjust the brightness of the current frame to clip bright pixels and threshold dark ones"""
        if self._min_value > self._max_value:
            self.current_frame = np.clip(self.current_frame, self._max_value, self._min_value) - self._max_value
            self.current_frame = 1. - self.current_frame / (self._min_value - self._max_value)
        else:
            self.current_frame = np.clip(self.current_frame, self._min_value, self._max_value) - self._min_value
            self.current_frame = self.current_frame / (self._max_value - self._min_value)

        self._mean_value = np.mean(self.current_frame)

    def adjust_avgframe(self):
        """Adjust the brightness of the average frame to clip bright pixels and threshold dark ones"""
        self.average_frame = np.clip(self.average_frame, self._min_value, self._max_value) - self._min_value
        self.average_frame = self.average_frame / (self._max_value - self._min_value)

    def subtract_avgframe(self, adjust=True):
        """"""
        self.current_frame = self.current_frame - self.average_frame

    def adjust_normbg(self):
        """Use self.normfactor_frame to mutliply the current frame by a map that attempts to equilize the brightness in
        different regions"""
        self.current_frame = self.current_frame * self.normfactor_frame

    def find_points_hough(self, roi=None):
        """Use Hough transform to find centers of circles (gyro edges)"""
        img = np.array(self.current_frame * 255, dtype=np.uint8)

        # apply blur so you don't find lots of fake circles
        img = cv2.GaussianBlur(img, (3, 3), 2, 2)

        if cv2.__version__[0] == '2':
            # Note: inputs for cv2.HoughCircles are:
            #     image	    8-bit, single-channel, grayscale input image.
            #     circles	Output vector of found circles. Each vector is encoded as a 3-element
            #               floating-point vector (x,y,radius) .
            #     method	Detection method, see cv::HoughModes. Currently, the only implemented
            #               method is HOUGH_GRADIENT
            #     dp	    Inverse ratio of the accumulator resolution to the image resolution.
            #               For example, if dp=1 , the accumulator has the same resolution as the input image.
            #               If dp=2 , the accumulator has half as big width and height.
            #     minDist	Minimum distance between the centers of the detected circles.
            #               If the parameter is too small, multiple neighbor circles may be falsely detected in
            #               addition to a true one. If it is too large, some circles may be missed.
            #     param1	First method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the higher
            #               threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
            #     param2	Second method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the accumulator
            #               threshold for the circle centers at the detection stage.
            #               The smaller it is, the more false circles may be detected.
            #               Circles, corresponding to the larger accumulator values, will be returned first.
            #     minRadius	Minimum circle radius.
            #     maxRadius	Maximum circle radius.
            circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, self.min_dist,
                                       param1=48, param2=18, minRadius=self.min_radius, maxRadius=self.max_radius)
        else:
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, self.min_dist,
                                       param1=48, param2=18, minRadius=self.min_radius, maxRadius=self.max_radius)

        circles = np.uint16(np.around(circles))
        self.circles = circles[0]
        xytmp = np.array([self.circles[:, 0], self.circles[:, 1]], dtype=float).T
        if roi is None:
            roi = self.roi

        if roi is not None:
            xy = dh.pts_in_polygon(xytmp, roi)
        else:
            xy = xytmp

        self.frame_current_points = xy

    # def find_points_houghmaskcentroid(self, roi=None, radius=10):
    #     """Use Hough transform to find centers of circles (where the circles are gyro edges), then move to the
    #     centroid within a circle with radius 'radius'"""
    #     img = np.array(self.current_frame * 255, dtype=np.uint8)
    #
    #     # apply blur so you don't find lots of fake circles
    #     img = cv2.GaussianBlur(img, (3, 3), 2, 2)
    #
    #     if cv2.__version__[0] == '2':
    #         # Note: inputs for cv2.HoughCircles are:
    #         #     image	    8-bit, single-channel, grayscale input image.
    #         #     circles	Output vector of found circles. Each vector is encoded as a 3-element
    #         #               floating-point vector (x,y,radius) .
    #         #     method	Detection method, see cv::HoughModes. Currently, the only implemented
    #         #               method is HOUGH_GRADIENT
    #         #     dp	    Inverse ratio of the accumulator resolution to the image resolution.
    #         #               For example, if dp=1 , the accumulator has the same resolution as the input image.
    #         #               If dp=2 , the accumulator has half as big width and height.
    #         #     minDist	Minimum distance between the centers of the detected circles.
    #         #               If the parameter is too small, multiple neighbor circles may be falsely detected in
    #         #               addition to a true one. If it is too large, some circles may be missed.
    #         #     param1	First method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the higher
    #         #               threshold of the two passed to the Canny edge detector (the lower one is twice smaller).
    #         #     param2	Second method-specific parameter. In case of CV_HOUGH_GRADIENT , it is the accumulator
    #         #               threshold for the circle centers at the detection stage.
    #         #               The smaller it is, the more false circles may be detected.
    #         #               Circles, corresponding to the larger accumulator values, will be returned first.
    #         #     minRadius	Minimum circle radius.
    #         #     maxRadius	Maximum circle radius.
    #         circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, self.min_dist,
    #                                    param1=48, param2=18, minRadius=self.min_radius, maxRadius=self.max_radius)
    #     else:
    #         circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, self.min_dist,
    #                                    param1=48, param2=18, minRadius=self.min_radius, maxRadius=self.max_radius)
    #
    #     circles = np.uint16(np.around(circles))
    #     self.circles = circles[0]
    #     xytmp = np.array([self.circles[:, 0], self.circles[:, 1]], dtype=float).T
    #     if roi is None:
    #         roi = self.roi
    #
    #     if roi is not None:
    #         xy = dh.pts_in_polygon(xytmp, roi)
    #     else:
    #         xy = xytmp
    #
    #     self.frame_current_points = xy

    def find_points_click(self, roi=None, tool='hough', pix=30,
                          image_kernel_path='./new_image_kern.png'):
        """Click on the positions of the points to get their positions manually
        Optionally use a tool transform to find approximate centers if kwarg tool is not None
        """
        # assume points are very roughly detected via hough trasnform or convolution if tool is not None
        if tool is None:
            # raise RuntimeError('Have not written yet: make a way to determine how many points to click (interactive)')
            self.frame_current_points = np.array([[np.shape(self.current_frame)[0] * 0.5,
                                                   np.shape(self.current_frame)[1] * 0.5]])
        elif tool == 'hough':
            self.find_points_hough(roi=roi)
        elif tool == 'convolution':
            self.find_points_convolution(image_kernel_path=image_kernel_path, roi=roi)
        elif tool == 'click':
            print 'movie_instance: click on initial finding points'
            impt = ImagePoint(img=self.current_frame)
            self.frame_current_points = np.array(impt.getCoord(figsize=(5, 5)))
            print 'self.frame_current_points = ', self.frame_current_points

        print 'len(np.shape(self.frame_current_points)) = ', len(np.shape(self.frame_current_points))
        if len(np.shape(self.frame_current_points)) == 1:
            self.frame_current_points = np.array([[self.frame_current_points[0], self.frame_current_points[1]]])

        new_points = []
        print 'movie_instance: np.shape(self.frame_current_points) = ', np.shape(self.frame_current_points)
        dmyk = 0
        for pta in self.frame_current_points:
            print 'click on pt ' + str(dmyk) + ' / ' + str(len(self.frame_current_points))
            print 'pta =', pta
            print 'pta[0] =', pta[0]
            print 'int(pta[0]) = ', int(pta[0])
            pt = [int(pta[0]), int(pta[1])]

            # w, h = np.shape(self.current_frame)
            # Center num_times in case the dot has moved partially out of the box during the step.
            # draw small boxes
            # print 'pt[1] - self._pix = ', (pt[1] - self._pix)
            # print 'pt[1] + self._pix = ', (pt[1] + self._pix)
            # print 'np.shape(self.current_frame) = ', np.shape(self.current_frame)
            bf = self.current_frame[int(pt[1] - pix):int(pt[1] + pix)]
            bf = bf[:, int(pt[0] - pix):int(pt[0] + pix)]
            # bf_comp = bf.copy()

            # let's clip this area to maximize the bright spot
            bf = bf.astype('f')

            print 'bf.flatten() = ', bf.flatten()
            bf_min = self._centroid_clipmin * np.max(bf.flatten())
            bf_max = self._centroid_clipmax * np.max(bf.flatten())
            print 'bf_min = ', bf_min
            print 'bf_max = ', bf_max
            bf = np.clip(bf, bf_min, bf_max) - bf_min
            bf /= (bf_max - bf_min)

            # Click on the correct position
            print 'Click on the correct position'
            impt = ImagePoint(img=bf)
            clickcenter = impt.getCoord(figsize=(5, 5))

            # find center of mass difference from center of box
            movx = - pix + clickcenter[0]  # pix - com[0]
            movy = - pix + clickcenter[1]  # pix - com[1]

            # move the points
            pt[0] = pt[0] + movx
            pt[1] = pt[1] + movy

            new_points.append(pt)
            dmyk += 1

        new_points = np.array(new_points, dtype=float)
        # print 'movie_instance: new_points = ', new_points
        ind = np.argsort(new_points[:, 0])
        new_points = new_points[ind]
        ind = np.argsort(new_points[:, 1])
        new_points = new_points[ind]

        # Check it
        # plt.clf()
        # plt.imshow(self.current_frame)
        # plt.plot(new_points[:, 0], new_points[:, 1], 'ro')
        # plt.show()
        # sys.exit()

        self.frame_current_points = np.array(new_points, dtype=float)
        return self.frame_current_points

    def find_points_convolution(self, image_kernel_path='./new_image_kern.png', roi=None):
        """Identify particles by convolving an image of a gyro over the movie image"""
        img = np.array(self.current_frame)
        # fig = plt.figure()
        # print 'movie_instance: showing figure to be convolved...'
        # plt.imshow(img, cmap=cm.Greys_r)
        # plt.show()

        img_ker = mpimg.imread(image_kernel_path)
        if len(np.shape(img_ker)) > 2:
            im2 = img_ker[:, :, 0:3]
            im3 = np.sum(im2, axis=-1)
            img_ker = im3
            im = Image.fromarray((img_ker * 255).astype(np.uint8))
            im.save(image_kernel_path.split('.png')[0] + '_flattened.png')

        print 'movie_instance: np.shape(img_ker) = ', np.shape(img_ker)

        img_ker[img_ker < 0.5] = -0.

        fr = ndimage.convolve(img, img_ker, mode='reflect', cval=0.0)
        minval = 0.0 * max(fr.flatten())
        maxval = 1. * max(fr.flatten())
        f = (np.clip(fr, minval, maxval) - minval) / (maxval - minval)

        data_max = filters.maximum_filter(f, 80)
        maxima = (f == data_max)

        data_min = filters.minimum_filter(f, 80)

        dmax = max((data_max - data_min).flatten())
        dmin = min((data_max - data_min).flatten())

        minmax = (dmax - dmin)

        diff = ((data_max - data_min) >= dmin + 0.10 * minmax)
        maxima[diff == 0] = 0

        labeled, num_object = ndimage.label(maxima)
        slices = ndimage.find_objects(labeled)

        x, y = [], []

        for dy, dx in slices:
            rad = np.sqrt((dx.stop - dx.start) ** 2 + (dy.stop - dy.start) ** 2)
            # print 'rad', rad
            if rad < 15 and rad > 0.25:
                # print ra
                x_center = (dx.start + dx.stop) / 2
                x.append(x_center)
                y_center = (dy.start + dy.stop) / 2
                y.append(y_center)

        xy = np.dstack((x, y))[0]
        # Check it
        print 'movie_instance: Showing the results of the convolution...'
        fig = plt.figure()
        plt.imshow(fr, cmap=cm.Greys_r)
        plt.plot(x, y, 'ro')
        plt.show()

        if roi is not None:
            xy = dh.pts_in_polygon(xy, roi)

        sys.exit()
        self.frame_current_points = xy
        return xy

    def click_reference_pts(self, pix=40):
        """Click points on a cirlce to get the positions of the reference points manually -- from a fit to the circle
        Optionally use a tool transform to find approximate centers if kwarg tool is not None
        """
        # assume points are very roughly detected via hough trasnform or convolution if tool is not None
        ref_points = []
        ii = 0
        for pta in self.frame_current_points:
            print 'determining mask for pt #' + str(ii) + ' / ' + str(len(self.frame_current_points))
            pt = [int(pta[0]), int(pta[1])]
            bf = self.current_frame[int(pt[1] - pix):int(pt[1] + pix)]
            bf = bf[:, int(pt[0] - pix):int(pt[0] + pix)]
            # bf_comp = bf.copy()

            # let's clip this area to maximize the bright spot
            bf = bf.astype('f')

            bf_min = self._centroid_clipmin * np.max(bf.flatten())
            bf_max = self._centroid_clipmax * np.max(bf.flatten())
            bf = np.clip(bf, bf_min, bf_max) - bf_min
            bf /= (bf_max - bf_min)

            # Click on four points on the enclosing circle
            clicks = []
            for dmyk in range(4):
                print 'click on pt #' + str(dmyk) + ' on the circle'
                impt = ImagePoint(img=bf)
                click = impt.getCoord(figsize=(5, 5))
                if None in click:
                    print 'click was off screen or no good, try again...'
                    click = impt.getCoord(figsize=(5, 5))
                clicks.append(click)

            clickcenter = np.mean(np.array(clicks), axis=0)
            print 'clicks = ', np.array(clicks)
            print 'GyroMovieclickcenter = ', clickcenter
            # sys.exit()

            # find center of mass difference from center of box
            movx = - pix + clickcenter[0]  # pix - com[0]
            movy = - pix + clickcenter[1]  # pix - com[1]

            # move the points
            pt[0] = pt[0] + movx
            pt[1] = pt[1] + movy

            ref_points.append(pt)
            ii += 1

        ref_points = np.array(ref_points, dtype=float)
        # print 'movie_instance: new_points = ', new_points
        ref_points = ref_points[np.argsort(ref_points[:, 0])]
        new_points = ref_points[np.argsort(ref_points[:, 1])]

        # Check it
        # plt.clf()
        # plt.imshow(self.current_frame)
        # plt.plot(new_points[:, 0], new_points[:, 1], 'ro')
        # plt.show()
        # sys.exit()

        self.reference_points = np.array(new_points, dtype=float)
        return self.reference_points

    # def center_on_bright_new(self, num_times):
    #     new_points = []
    #
    #     for pt in self.frame_current_points:
    #
    #         h, w = np.shape(self.current_frame)
    #         # if ((pt[0] > 1.5 * self._pix) and (pt[1] > 1.5 * self._pix) and (pt[0] < w - 1.5 * self._pix) and (
    #         #   pt[1] < h - 1.5 * self._pix)):
    #         if True:
    #             for j in xrange(num_times):
    #                 # Center num_times in case the dot has moved partially out of the box during the step.
    #                 # draw small boxes
    #                 bf = self.current_frame[pt[1] - self._pix:pt[1] + self._pix]
    #                 bf = bf[:, pt[0] - self._pix:pt[0] + self._pix]
    #                 bf_comp = bf.copy()
    #                 # let's clip this area to maximize the bright spot
    #                 bf = bf.astype('f')
    #
    #                 bf_min = 0.8 * np.max(bf.flatten())
    #                 bf_max = 1. * np.max(bf.flatten())
    #                 bf = np.clip(bf, bf_min, bf_max) - bf_min
    #                 bf = bf / (bf_max - bf_min)
    #                 bf = cv2.GaussianBlur(bf, (2, 2), 1, 1)
    #
    #                 # find center of brightness
    #                 data_max = filters.maximum_filter(bf, self._pix)
    #                 data_min = filters.minimum_filter(bf, self._pix)
    #                 maxima = (bf == data_max)
    #                 dmax = max((data_max - data_min).flatten())
    #                 dmin = min((data_max - data_min).flatten())
    #                 minmax = (dmax - dmin)
    #                 diff = ((data_max - data_min) >= dmin + 0.9 * minmax)
    #                 maxima[diff == 0] = 0
    #                 maxima = (bf == data_max)
    #
    #                 labeled, num_object = ndimage.label(maxima)
    #                 slices = ndimage.find_objects(labeled)
    #
    #                 x, y = [], []
    #                 for dx, dy in slices:
    #                     rad = np.sqrt((dx.stop - dx.start) ** 2 + (dy.stop - dy.start) ** 2)
    #                     if rad < 3:
    #                         x_center = (dx.start + dx.stop) / 2
    #                         x.append(x_center)
    #                         y_center = (dy.start + dy.stop) / 2
    #                         y.append(y_center)
    #                 com = [x[0], y[0]]
    #
    #                 # find center of mass difference from center of box
    #                 movx = self.dummy[1] - com[1]  # pix - com[0]
    #                 movy = self.dummy[0] - com[0]  # pix - com[1]
    #
    #                 if math.isnan(movx):
    #                     movx = 0
    #                 if math.isnan(movy):
    #                     movy = 0
    #
    #                 # move the points
    #                 pt[0] = pt[0] - movx
    #                 pt[1] = pt[1] - movy
    #
    #                 if j == num_times - 1:
    #                     fig = plt.figure()
    #                     plt.imshow(bf)
    #
    #                     plt.plot(pt[0], pt[1], 'ro')
    #                     plt.show()
    #
    #             if np.mean(bf_comp) < 5 * self._mean_value:
    #                 new_points.append(pt)
    #
    #     new_points = np.array(new_points, dtype=float)
    #     ind = np.argsort(new_points[:, 0])
    #     new_points = new_points[ind]
    #     ind = np.argsort(new_points[:, 1])
    #     new_points = new_points[ind]
    #
    #     self.frame_current_points = np.array(new_points, dtype=float)

    def center_on_bright(self, num_times=2, pix=None):
        """Given current frame points, find centroid of brightness and replace current frame points with new centroid
        points. Updates self.frame_current_points

        Parameters
        ----------
        num_times : int
            How many times to iterate finding the centroid in pix x pix subset of the image,
            (ie find centroid, move current points to that site), then repeat num_times - 1 times.
        pix : int or None
            size of each window to make in order to find the centroid of brightness in the vecinity of the approximate
            position given by self.frame_current_points.
            If None, uses self._pix
        """
        new_points = []

        if pix is None:
            pix = self._pix

        # print 'movie_instance: self.frame_current_points = ', self.frame_current_points
        for pta in self.frame_current_points:
            pt = [int(pta[0]), int(pta[1])]

            # w, h = np.shape(self.current_frame)
            # Previously, there was a condition here on the position to add the point
            # if ((pt[0] > 1.5*self._pix) and (pt[1] > 1.5*self._pix) and (pt[0] < w - 1.5*self._pix) and
            # (pt[1] < h - 1.5*self._pix)):
            for jj in xrange(num_times):
                # Center num_times in case the dot has moved partially out of the box during the step.
                # draw small boxes

                # print 'num_times = ', num_times
                # print 'pt[1] - self._pix = ', (pt[1] - self._pix)
                # print 'pt[1] + self._pix = ', (pt[1] + self._pix)
                # print 'np.shape(self.current_frame) = ', np.shape(self.current_frame)
                bf = self.current_frame[int(pt[1] - pix):int(pt[1] + pix)]
                bf = bf[:, int(pt[0] - pix):int(pt[0] + pix)]
                # bf_comp = bf.copy()

                # let's clip this area to maximize the bright spot
                bf = bf.astype('f')

                bf_min = self._centroid_clipmin * np.max(bf.flatten())
                bf_max = self._centroid_clipmax * np.max(bf.flatten())
                bf = np.clip(bf, bf_min, bf_max) - bf_min
                bf /= (bf_max - bf_min)

                # Set all pixels further than pix from the center to dark --> the mask is a circle
                yy, xx = np.ogrid[-pix + 0.5:pix + 0.5, -pix + 0.5:pix + 0.5]
                mask = xx * xx + yy * yy > pix * pix
                # print 'movie_instance: mask = ', mask
                # print 'np.shape(mask) = ', np.shape(mask)
                # print 'np.shape(bf) = ', np.shape(bf)
                if np.shape(mask)[0] < np.shape(bf)[0] and np.shape(mask)[1] < np.shape(bf)[1]:
                    bf[mask] = 0

                # # Check it
                # plt.clf()
                # plt.imshow(bf)
                # plt.show()
                # sys.exit()

                # find center of brightness
                com = ndimage.measurements.center_of_mass(bf)

                # Check
                # if jj == num_times -1:
                #     fig = plt.figure()
                #     plt.imshow(bf)
                #     plt.show()

                # find center of mass difference from center of box
                movx = self.dummy[1] - com[1]  # pix - com[0]
                movy = self.dummy[0] - com[0]  # pix - com[1]

                if math.isnan(movx):
                    movx = 0
                if math.isnan(movy):
                    movy = 0

                # move the points
                pt[0] = pt[0] - movx
                pt[1] = pt[1] - movy

            new_points.append(pt)

        new_points = np.array(new_points, dtype=float)
        # print 'movie_instance: new_points = ', new_points
        ind = np.argsort(new_points[:, 0])
        new_points = new_points[ind]
        ind = np.argsort(new_points[:, 1])
        new_points = new_points[ind]

        self.frame_current_points = np.array(new_points, dtype=float)

    def center_on_bright_difference(self, num_times=2, check=False):
        """Given current frame points, find centroid of brightness difference between current frame and previous frame
        and replace current frame points with new points from the centroid of diff(frames)"""
        new_points = []

        for pta in self.frame_current_points:
            pt = [int(pta[0]), int(pta[1])]

            # w, h = np.shape(self.current_frame)
            # Previously, there was a condition here on the position to add the point
            # if ((pt[0] > 1.5*pix) and (pt[1] > 1.5*pix) and (pt[0] < w - 1.5*pix) and
            # (pt[1] < h - 1.5*pix)):
            for jj in xrange(num_times):
                # Center num_times in case the dot has moved partially out of the box during the step.
                # draw small boxes

                # print 'num_times = ', num_times
                # print 'pt[1] - pix = ', (pt[1] - pix)
                # print 'pt[1] + pix = ', (pt[1] + pix)
                # print 'np.shape(self.current_frame) = ', np.shape(self.current_frame)
                bf = self.current_frame[int(pt[1] - self._pix):int(pt[1] + self._pix),
                                        int(pt[0] - self._pix):int(pt[0] + self._pix)]
                if not self.previous_frame is None:
                    bfprev = self.previous_frame[int(pt[1] - self._pix):int(pt[1] + self._pix),
                                                 int(pt[0] - self._pix):int(pt[0] + self._pix)]
                    bf1 = bf - bfprev
                else:
                    bf1 = bf

                # let's clip this area to maximize the bright spot
                bf2 = bf1.astype('f')

                bf_min = self._centroid_clipmin * np.max(bf2.flatten())
                bf_max = self._centroid_clipmax * np.max(bf2.flatten())
                bf2 = np.clip(bf2, bf_min, bf_max) - bf_min
                bf2 /= (bf_max - bf_min)

                # find center of brightness
                com = ndimage.measurements.center_of_mass(bf2)

                # Check the tracking
                if check:
                    if not self.previous_frame is None:
                        if jj == num_times -1:
                            fig = plt.figure()
                            ax1 = fig.add_subplot(221)
                            ax2 = fig.add_subplot(222)
                            ax3 = fig.add_subplot(223)
                            ax4 = fig.add_subplot(224)
                            ax1.imshow(bf)
                            ax2.imshow(bfprev)
                            ax3.imshow(bf1)
                            im = ax4.imshow(bf2)
                            fig.colorbar(im)
                            plt.pause(0.001)
                            plt.close('all')

                # find center of mass difference from center of box
                movx = self.dummy[1] - com[1]  # pix - com[0]
                movy = self.dummy[0] - com[0]  # pix - com[1]

                if math.isnan(movx):
                    movx = 0
                if math.isnan(movy):
                    movy = 0

                # move the points
                pt[0] = pt[0] - movx
                pt[1] = pt[1] - movy

            new_points.append(pt)

        new_points = np.array(new_points, dtype=float)
        ind = np.argsort(new_points[:, 0])
        new_points = new_points[ind]
        ind = np.argsort(new_points[:, 1])
        new_points = new_points[ind]

        self.frame_current_points = np.array(new_points, dtype=float)

    def update_previous_frame(self):
        self.previous_frame = copy.deepcopy(self.current_frame)

    def save_frame(self, name='frame'):
        fig = plt.figure()
        ax = plt.axes([0, 0, 1, 1])
        img = cine.asimage(self.current_frame)
        plt.imshow(img, cmap=cm.Greys_r)
        plt.savefig(name + '.png')
        plt.close()

    def save_variance_frame(self, name='variance_frame'):
        fig, ax, cax = leplt.initialize_1panel_cbar_cent(Wfig=180, Hfig=200, wsfrac=1.0)
        # img = cine.asimage(self.variance_frame)
        img = self.variance_frame
        ax.imshow(img, cmap=cm.Greys_r, vmin=0., vmax=np.mean(self.variance_frame.ravel()))
        # Set color limits for cbar
        sm = leplt.empty_scalar_mappable(vmin=0., vmax=np.mean(self.variance_frame.ravel()), cmap=cm.Greys_r)
        plt.colorbar(sm, cax=cax, orientation='horizontal')
        plt.savefig(name + '.png')
        print 'img = ', img
        print 'np.min(img) = ', np.min(img)
        print 'np.max(img) = ', np.max(img)
        sys.exit()
        plt.close()

    def save_maxdiff_frame(self, name='maxdiff_frame'):
        fig, ax, cax = leplt.initialize_1panel_cbar_cent(Wfig=180, Hfig=200, wsfrac=1.0)
        # img = cine.asimage(self.variance_frame)
        img = self.maxdiff_frame
        ax.imshow(img, cmap=cm.Greys_r, vmin=0., vmax=np.mean(self.maxdiff_frame.ravel()))
        # Set color limits for cbar
        sm = leplt.empty_scalar_mappable(vmin=0., vmax=np.mean(self.maxdiff_frame.ravel()), cmap=cm.Greys_r)
        plt.colorbar(sm, cax=cax, orientation='horizontal')
        plt.savefig(name + '.png')
        print 'img = ', img
        print 'np.min(img) = ', np.min(img)
        print 'np.max(img) = ', np.max(img)
        sys.exit()
        plt.close()

    def return_image_data(self):
        return cine.asimage(self.current_frame)

    def save_frame_with_boxes(self, name='frame', pix=None):
        fig = plt.figure()
        ax = plt.axes([0, 0, 1, 1])
        img = np.array(self.current_frame)

        # get the pixel size used to find centroids if not supplied
        if pix is None:
            pix = self._pix

        # Store the maximum brightness in the image
        imgmax = np.max(img.ravel())

        if len(np.shape(self.frame_current_points)) == 1:
            self.frame_current_points = self.frame_current_points.reshape(1, -1)
        for pt in self.frame_current_points:
            pt = [int(pt[0]), int(pt[1])]
            # Invert the part inside the centroid capturing region
            # First grab the square
            boximg = img[pt[1] - pix:pt[1] + pix, pt[0] - pix: pt[0] + pix]
            # Now invert within the centroid-finding radius
            # Set all pixels further than pix from the center to dark
            yy, xx = np.ogrid[-pix + 0.5:pix + 0.5, -pix + 0.5:pix + 0.5]
            mask = xx * xx + yy * yy <= pix * pix
            boximg[mask] = imgmax - boximg[mask]
            img[pt[1] - pix:pt[1] + pix, pt[0] - pix: pt[0] + pix] = boximg

        # cine.asimage(img).save('image_kernel.png')
        img = cine.asimage(img)

        plt.imshow(img, cmap=cm.Greys_r)

        plt.savefig(name + '.png')
        plt.close()

    def find_point_convolve(self, img_ker):
        fr = ndimage.convolve(self.current_frame, img_ker, mode='reflect', cval=0.0)
        minval = 0.1 * max(fr.flatten())
        maxval = 1 * max(fr.flatten())
        fr = (np.clip(fr, minval, maxval) - minval) / (maxval - minval)

        fig = plt.figure()
        plt.imshow(fr)
        plt.show()

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])

