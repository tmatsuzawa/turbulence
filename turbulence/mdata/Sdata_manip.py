import os.path
import turbulence.manager.file_architecture as file_architecture
import turbulence.tools.pickle_m as pickle_m
import turbulence.tools.browse as browse
from turbulence.mdata.Sdata import Sdata
import turbulence.tools.rw_data as rw_data
import turbulence.tools.pickle_2json as p2json

"""
"""


# Sdata_gen_day is called by Data_load
def Sdata_gen_day(date):
    # find the location of the data for a given date
    fileDir = file_architecture.get_dir(date) + '/'

    print 'Name of Directory Being Processed: ' + fileDir
    #    print(fileDir)
    cineList, n = browse.get_fileList(fileDir, 'cine', display=True)
    print cineList

    Slist = []  #Why is this called "failure"????? -tmatsu
    for name in cineList:
        S = Sdata_gen(name, cineList.index(name))
        if not S is None:
            Slist.append(S)


    n = len(cineList)
    dict_day = {'Names': cineList, 'Index': range(n)}
    print(dict_day)  #print cine filenames and associated indicies
    filename = fileDir + 'Sdata_' + date + '/Cine_index_' + date + '.txt'
    rw_data.write_a_dict(filename, dict_day)

    return Slist


def Sdata_gen(cineFile, index):  #Called by
    """

    Parameters
    ----------
    cineFile : string
               name of a cine file
    index : intege
            index is associated to a specific cine file
    Returns
    S : Sdata class object
    -------
    """
    failure = None
    # print(cineFile)

    # find the file where exp. parameters are given
    base = browse.get_string(os.path.basename(cineFile), '', end='.cine') # Get the name of cineFile like "awesome_video" from "//path/to/.../awesome_video.cine"

    fileDir = os.path.dirname(cineFile) + '/'

    fileList, n = browse.get_fileList(fileDir, 'txt', root='Setup_file_Reference_', display=True, sort='date') # Get the Setup_file(s) from the directory, n is the number of Setup files

    file_param = os.path.dirname(cineFile) + '/' + 'Setup_file_Ref.txt'  # path of the file where relevant experimental parameters such as fx and sample rate is stored.

    print '----Setup txt file(s)---'
    print fileList
    print '------------------------'


    for f in fileList:
        s = browse.get_string(f, 'Setup_file_Reference_', end='.txt')  # name of Setup txt file
        if browse.contain(base, s): # if cineFile is Setup txt file where experimental parameters are stored, True
            file_param = f
            # input()
            #    file = os.path.dirname(cineFile)+'/References/Ref_'+base+'.txt'
            #    file = os.path.dirname(cineFile)+'/'+'Setup_file_Ref.txt' #/Volumes/labshared/Stephane_lab1/25015_09_22/References/Ref_PIV_sv_vp_zoom_Polymer_200ppm_X25mm_fps5000_n18000_beta500mu_H1180mm_S300mm.txt'
            #    print(fitule)
            # try:
    print '----Cine file being processed:----'
    print cineFile
    print '----------------------------'

    # S is a S_data class object which contains Id, param,fileDir,fileCine,dirCine.
    S = Sdata(fileCine=cineFile, index=index, fileParam=file_param)  # from turbulence.mdata.Sdata
    print '----------------------------'
    print 'Sdata class object S is created!'
    print 'Attributes of S are...'
    print dir(S)
    print 'Attributes of S.param are...'
    print dir(S.param)
    print '----------------------------'

    S.write()
    return S


def load_Sdata_day(date):
    # load all the data of the day, in a given Slist
    Dir = getDir(date)
    # print(Dir)
    fileList, n = browse.get_fileList(Dir, 'hdf5', display=True)

    if n > 0:
        Slist = []
        for filename in fileList:
            S = load_Sdata_file(filename)
            if S is not None:
                Slist.append(S)
        return Slist
    else:
        print("No hdf5 files found at " + Dir)


def load_Sdata_file(filename):
    # load a Sdata using its filename
    #  print(filename)
    if os.path.isfile(filename):
        #  print('Reading Sdata')
        S = Sdata(generate=False)  # create an empty instance of Sdata object
        setattr(S, 'filename', filename[:-5])  # set the reference to the filename where it was stored
        # print(S.filename)
        S.load_all()  # load all the parameters (only need an attribute filename in S)

        # print(S.fileCine)
        return S
    else:
        print('No data found for ' + S.filename)
        return None


def load_serie(date, indices):
    n = len(indices)
    Slist = [None for i in range(n)]
    c = 0
    for i in indices:
        Slist[c] = load_Sdata(date, i)
        c += 1
    return Slist


def load_measures(Slist, indices=0):
    """
    Load the measures associated to each element of Slist.
    by default, only load the first set.
    if indices = None, load all the Measures associated to each Sdata
    """

    Mlist = []
    for S in Slist:
        output = S.load_measures()
        # sort ouput by length
        # print(output)
        output = sorted(output, key=lambda s: (s.shape()[2], s.shape()[1]))
        print(output)
        if indices is None:
            Mlist.append(output)
        else:
            if not output == []:
                Mlist.append(output[indices])
            else:
                print('Sdata unprocessed')
                Mlist.append(None)
    return Mlist


def load_all():
    # for all the subdir in rootDir, find all the cine files and their associated Sdata.
    # return a List of all Sdata
    Slist = []
    for rootDir in file_architecture.Dir_data:
        dataDirList, n = browse.get_dirList(rootDir, '??*??', True)

        for Dir in dataDirList:
            date = browse.get_end(Dir, rootDir)
            Slist = Slist + load_Sdata_day(date)

    return Slist


def load_Sdata(date, index, mindex=0):
    # load a Sdata using its Id
    filename = getloc(date, index, mindex)
    S = load_Sdata_file(filename)
    return S


def read(filename, data=False):
    S = pickle_m.read(filename)

    if data:
        S.load_measure()
    return S


def getloc(date, index, mindex, frmt='.hdf5'):
    Dir = getDir(date)
    filename = Dir + "Sdata_" + date + "_" + str(index) + "_" + str(mindex) + frmt
    return filename


def getloc_S(S, frmt='.hdf5'):
    Dir = getDir(S.id.date)
    filename = Dir + "Sdata_" + S.id.date + "_" + str(S.id.index) + "_" + str(S.id.mindex) + frmt
    return filename


def getDir(date):
    rootdir = file_architecture.get_dir(date)
    Dir = rootdir + "/Sdata_" + date + "/"
    return Dir


def main():
    # date='2015_03_24'
    Sdata_gen_day(date)
    Measure_gen_day(date)
