# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:42:10 2015

@author: stephane
"""
import turbulence.tools.browse as browse
import turbulence.hdf5.h5py_convert as to_hdf5
import h5py
"""
Id class has the following attributes: index, mindex=0, type, and who
Currently, Sdata class has an attribute called "Id" which belongs to this Id class.
index: index associated to PIV data of a cine file
mindex=0: index for mdata... index of mdata is necessary apart from index because "index" is associated to a cine file, but different mdata can be generated from the same cine file by changing PIV settings.
type: experimental type, default is vortex collision
who: conductor of the experiment
"""

class Id:
    def __init__(self, S=None, generate=True, **kwargs):
        # list of parameters, should be a dictionnary ?
        # Identification number
        self.Sdata = S

        if generate:
            self.gen(**kwargs)

    def __name__(self):
        return 'Id'

    def load(self, data):
        self = to_hdf5.load(self, data)

    def gen(self, index=-1, mindex=0, type="vortex collision", who='takumi'):
        self.index = index
        self.mindex = mindex
        self.type = type
        self.who = who

        if hasattr(self.Sdata, 'fileCine'):
            self.set_date(self.Sdata.fileCine)
        else:
            self.date = ''

    def get_id(self):
        identifiant = self.date + "_" + str(self.index) + "_" + str(self.mindex)
        return identifiant

    def set_date(self, file):
        # Date
        #        file=self.Sdata.fileCine
        self.date = browse.get_string(file, "/20", "/", -2)
