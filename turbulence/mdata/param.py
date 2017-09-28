# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:42:10 2015

@author: stephane
"""

import os.path
import matplotlib.pyplot as plt
import numpy as np

import turbulence.cine as cine
import turbulence.mdata.initial_time as initial_time
import turbulence.tools.rw_data as rw_data
import turbulence.tools.browse as browse

import turbulence.hdf5.h5py_convert as to_hdf5
import h5py

import glob
"""
param Class has the following attributes:
Xplane: ?
Zplane: ?
fps: sample rate
H0: ?
stroke: ?
typeview: Camera view- Possible values are... sv (side view), fv(front view), bv(????), and NA (not available)
typeplane: Plane of the laser sheet- Possible values are... hp(horizontal plane) and vp(vertical plane)
im0: frame number that you would like to start generating data file (.hdf5)
"""

class param:
    def __init__(self, S=None, generate=True):
        self.Sdata = S

        if generate:
            self.fileParam = self.filename_param('r')  #self.filename_param('r')  outputs filename
    def __name__(self):
        return 'param'

        # Experimental condition

    def load(self, data):
        self = to_hdf5.load(self, data)

    def set_param(self):  #Pick up experimental parameters from Cine filename
        #Sample fileCine is "PIV_sv_vp_left_100mmzeiss_fps3000_8holes_D20mm_span10000micron_freq5Hz_v400mms_1"
        file = self.Sdata.fileCine

        self.Xplane = browse.get_number(file, "_X", "mm", from_end=False)
        self.Zplane = browse.get_number(file, "_Z", "mm", from_end=False)
        self.fps = browse.get_number(file, "_fps", "_", from_end=False)
        self.H0 = browse.get_number(file, "_H", "mm", from_end=False)
        self.stroke = browse.get_number(file, "_S", "mm", from_end=False)

        if file.find('_sv') > 0:
            self.typeview = 'sv'    #side view
        elif file.find('_fv') > 0:
            self.typeview = 'fv'  #front view
        elif file.find('bv'):
            self.typeview = 'bv' #bv...?
        else:
            self.typeview = 'NA'

        if file.find('_hp') > 0:   #horizontal plane
            self.typeplane = 'hp'
        else:
            if file.find('_vp') > 0:  #vertical plane
                self.typeplane = 'vp'
            else:
                self.typeplane = 'NA'


        if (not hasattr(self, 'im0')) or (self.im0 == 0):
            pass
            #     self.get_im0()

    def load_exp_parameters(self, fileParam=''):
        """
        Read experimental parameters from a setup file, or let user manually type them on the screen.
        The parameters are stored in a param class object, and this param class object is attributed to a Sdata class object.
        Parameters
        ----------
        opt : str, default: "w"
            option. If opt=="r", the filename becomes /path/to/.../CineFile_param   . (Extension is not included)
            Otherwise, the filename becomes /path/to/../CineFileDir/PIV_date_index_mindex_param

        Returns
        name : str
            "/.../CineFileDir/CineFile_param" or "/.../CineFileDir/PIV_date_index_mindex_param"
        -------
        """

        self.fileParam = self.fileParam + '.txt'
        self.im_ref = self.read_reference(ref=0)   #self.read_reference(ref=0) outputs spatial resolution of cine images. e.g.- (640, 1024) Note that this method uses the cine Class.
        # default value of fileParam :
        if True:  # fileParam == '':
            base = browse.get_string(os.path.basename(self.Sdata.fileCine), '', end='.cine')
            #   print(base)
            fileRef = os.path.dirname(self.Sdata.fileCine) + '/Setup_file_Reference*.txt'
            #   print(fileRef)
            fileList = glob.glob(fileRef)     # make a list of Setup_file_Reference*.txt files.

            #print(base)
            #print(fileList)
            for filename in fileList:
                pattern = browse.get_string(filename, 'Setup_file_Reference', end='.txt')
                if pattern in base:  # if there is such a txt file with a filename which contains 'Setup_file_Reference', the file must be the one that cotains experimental parameters.
                    fileParam = filename

        # other name :
        print 'Experimental parameters may be stored here:'
        print fileParam
        print 'Does this file exist?'
        if os.path.isfile(fileParam):
            print '... Yes! Read experimental parameters from this file.'
            #Read the parameters from the param file
            Header, Data = rw_data.read_dataFile(fileParam, Hdelimiter='\t', Ddelimiter='\t')
#            print Header
            print Data

            #Add the Data keys to attributes
            for i in Data.keys():
                setattr(self, i, Data[i][0])
            print 'Now, this param class object has new attributes which are named as same as the header of ' + fileParam

        else:
            print '... No. You specified to read ' + self.fileParam + ' if this is just ".txt", you did not specify the setup file. '
            print 'Does this file actually exist?'
            ######## If the argument "fileParam" is given,... read that file as a setup file.
            if os.path.isfile(self.fileParam):
                self.im_ref = self.read_reference(ref=0)  #Read spatial resolution of cine images directory (i.e.- Open the target cine file, and grab the spatial resolution information)
                self.Sdata.read_param()     # load parameters from the parameter file
            #######
            ####### If the algorithm above could not find the setuo file, manually input the experimental parameters.
            else:
                print '... No. '
                print 'We could not find the setup file. Please type experimental parameters now.'
                self.im_ref = self.read_reference(ref=0)  #Read spatial resolution of cine images directory (i.e.- Open the target cine file, and grab the spatial resolution information)
                self.manual_input()

    # self.generate_parameter_file()

    def filename_param(self, opt="w"):
        """
        Generates a filename called "/.../CineFileDir/CineFile_param" or "/.../CineFileDir/PIV_date_index_mindex_param"
        Parameters
        ----------
        opt : str, default: "w"
            option. If opt=="r", the filename becomes /path/to/.../CineFile_param   . (Extension is not included)
            Otherwise, the filename becomes /path/to/../CineFileDir/PIV_date_index_mindex_param

        Returns
        name : str
            "/.../CineFileDir/CineFile_param" or "/.../CineFileDir/PIV_date_index_mindex_param"
        -------
        """
        if opt == "r":
            name = self.Sdata.fileCine[:-5] + "_param"  # without the cine extension
        else:
            #            print(self.Sdata.fileCine)
            name = "PIV_" + self.Sdata.get_id() + "_param"
            name = self.Sdata.dirCine + name
        return name

    def manual_input(self):
        # load the first image and save it in a reference file
        print(
        "Manual loading of experimental parameters for " + self.Sdata.fileCine)  # self.date + " ," + str(self.fps) +" fps")
        self.fx = float(input("fx = "))
        self.im0 = float(input("im(t=0) = "))
        self.x0 = float(input("x0 = "))
        self.y0 = float(input("y0 = "))
        self.angle = float(input("angle = "))
        # generate a param file (in case of the data have to be reloaded)
        # name of the file :
        # Rotating angle

    def get_im0(self):
        self.im0 = initial_time.get_im0(self, int(self.im0))

    def comments(self):
        self.description = input("")

    def read_reference(self, ref=0):
        """
        Returns the grayscale values (???) from a certain frame of a cine file specified by "ref"
        ----------
        ref : int, default=0

        Returns
        im_ref : numpy float64 matrix
            This matrix stores a sample grayscale values(???) from a certain frame of a cine file specified by "ref".
            This method also converts the values for bpp = 8.
            np.shape(im_ref) gives a spatial resolution of the cine images e.g.- (624,1024)
        -------
        """

        print 'Cine file being processed:' + self.Sdata.fileCine
        try:
            c = cine.Cine(self.Sdata.fileCine)
        except:
            print('Cine file unavailable')
            return None
        print('cine open')
        # get maximum value according to bit depth
        bitmax = float(2 ** c.real_bpp) #bpp: bits per pixel


        # get frames from cine file
        a = c[ref].astype("f")  # "f" is float
        a = a / bitmax * 255    #0-255: 8 bits   # Now, each element of a can be expressed by 8bits.

        im_ref = np.float64(a) #Then, use float64 (Double precision float: sign bit, 11 bits exponent, 52 bits mantissa)o store a, name it im_ref

        print 'Spatial resolution of the cine images: ' +  str(np.shape(im_ref)) + ' in px'  # Note np.shape(im_ref) spits out the spatial resolution. eg. (624 px,1024 px)

        # plt.figure(1)
        # plt.imshow(im_ref)
        # #        graphes.refresh()
        # plt.show(True)
        # # save the figure
        # plt.draw()

        c.close()

        print('cine closed')

        return im_ref
