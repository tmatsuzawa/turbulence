import os.path
import turbulence.manager.file_architecture as file_architecture
import turbulence.tools.pickle_m as pickle_m
import turbulence.tools.browse as browse
from turbulence.mdata.Sdata import Sdata
import turbulence.tools.rw_data as rw_data
import turbulence.tools.pickle_2json as p2json

"""
"""


def Sdata_gen_day(date):
    # find the location of the data for a given date
    fileDir = file_architecture.get_dir(date) + '/'

    print(fileDir)
    #    print(fileDir)
    cineList, n = browse.get_fileList(fileDir, 'cine', display=True)
    # cineList=
    # ['/Volumes/labshared3/Stephane/Experiments/2015_03_04/PIV_sv_X25mm_Z150mm_fps10000_H1000mm_zoom_S100mm.cine',
    # '/Volumes/labshared3/Stephane/Experiments/2015_03_04/PIV_sv_X25mm_Z200mm_fps10000_H1000mm_zoom_S100mm.cine',
    # '/Volumes/labshared3/Stephane/Experiments/2015_03_04/PIV_sv_X25mm_Z0mm_fps10000_H1380mm_zoom_zoom_S100mm.cine']
    # cineList=
    # ['/Volumes/labshared3/Stephane/Experiments/2015_03_04/PIV_sv_X25mm_Z0mm_fps10000_H1380mm_zoom_zoom_S100mm.cine']
    # cineList=['/Volumes/labshared3/Stephane/Experiments/2015_03_04/PIV_sv_X0mm_fps10000_H1000mm_zoom_S100mm.cine']
    print(cineList)
    failure = []
    for name in cineList:
        #  print(name)
        output = Sdata_gen(name, cineList.index(name))
        if not output is None:
            failure.append(output)

    n = len(cineList)
    dict_day = {'Names': cineList, 'Index': range(n)}
    print(dict_day)
    filename = fileDir + 'Sdata_' + date + '/Cine_index_' + date + '.txt'
    rw_data.write_a_dict(filename, dict_day)

    print(failure)
    return failure


def Sdata_gen(cineFile, index):
    """

    Parameters
    ----------
    cineFile
    index

    Returns
    -------
    S :
    """
    failure = None
    # print(cineFile)
    base = browse.get_string(os.path.basename(cineFile), '', end='.cine')
    # find the file where exp. parameters are given

    fileDir = os.path.dirname(cineFile) + '/'

    fileList, n = browse.get_fileList(fileDir, 'txt', root='Setup_file_Reference_', display=True, sort='date')

    file_param = os.path.dirname(cineFile) + '/' + 'Setup_file_Ref.txt'

    print(fileList)
    print('toto')
    for f in fileList:
        s = browse.get_string(f, 'Setup_file_Reference_', end='.txt')
        # print(base)
        # print('extract :'+s)
        if browse.contain(base, s):
            file_param = f
            # input()
            # file = os.path.dirname(cineFile)+'/References/Ref_'+base+'.txt'
            # file = os.path.dirname(cineFile)+'/'+'Setup_file_Ref.txt'
            # #/Volumes/labshared/Stephane_lab1/25015_09_22/References/Ref_PIV_sv_vp_zoom_Polymer_200ppm_X25mm_fps5000_n18000_beta500mu_H1180mm_S300mm.txt'
            #    print(file)
    print(cineFile)
    S = Sdata(fileCine=cineFile, index=index, fileParam=file_param)
    S.write()
    return S


def load_Sdata_day(date):
    # load all the data of the day, in a given Slist
    Dir = getdir(date)
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
    """load a Sdata using its filename

    Parameters
    ----------
    filename : str
        load the data from the given filename

    Returns
    -------
    ss : Sdata object or None
    """
    #  print(filename)
    if os.path.isfile(filename):
        #  print('Reading Sdata')
        # todo: Erase instantiation without __init__ of attributes here
        ss = Sdata(generate=False)  # create an empty instance of Sdata object
        setattr(ss, 'filename', filename[:-5])  # set the reference to the filename where it was stored
        print('Sdata_manip.load_Sdata_file(): ss.filename = ' + ss.filename)
        ss.load_all()  # load all the parameters (only need an attribute filename in S)

        print('Sdata_manip.load_Sdata_file(): ss.fileCine = ' + ss.fileCine)
        return ss
    else:
        print('No data found for ' + filename)
        return None


def load_serie(date, indices, rootdir=None):
    """Load a list of Sdata objects

    Parameters
    ----------
    date :
    indices : list of ints
        The indices of the cine files (I think?) found in the directory which matches the specified date

    Returns
    -------
    Slist : list of Sdata objects

    """
    n = len(indices)
    Slist = [None for i in range(n)]
    c = 0
    for i in indices:
        Slist[c] = load_Sdata(date, i, rootdir=rootdir)
        c += 1
    return Slist


def load_measures(Slist, indices=0):
    """Load the measures associated to each element of Slist.
    By default, only load the first set.
    If indices = None, load all the Measures associated to each Sdata

    Parameters
    ----------
    Slist : list of ...?
    indices : int or list of ints
        The indices of the Slist for which to load measures
    """
    Mlist = []
    for S in Slist:
        print 'Sdata_manip.load_measures(): Slist[current] = ', S
        output = S.load_measures()
        # sort ouput by length
        # print(output)
        output = sorted(output, key=lambda s: (s.shape()[2], s.shape()[1]))
        #  print(output)
        if indices is None:
            Mlist.append(output)
        else:
            if not output == []:
                print('Sdata_manip.load_measures(): added mdata for Sdata element in Slist')
                Mlist.append(output[indices])
            else:
                print('Sdata_manip.load_measures(): Sdata unprocessed')
                Mlist.append(None)
    return Mlist


def load_all():
    """For all the subdir in rootDir, find all the cine files and their associated Sdata.
    Return a List of all Sdata

    Returns
    -------
    Slist : list of all Sdata for all cine files
    """
    # for all the subdir in rootDir, find all the cine files and their associated Sdata.
    # return a List of all Sdata
    Slist = []
    for rootDir in file_architecture.Dir_data:
        dataDirList, n = browse.get_dirList(rootDir, '??*??', True)

        for Dir in dataDirList:
            date = browse.get_end(Dir, rootDir)
            Slist = Slist + load_Sdata_day(date)

    return Slist


def load_Sdata(date, index, mindex=0, rootdir=None):
    """Load

    Parameters
    ----------
    date
    index
    mindex
    datadir : str

    Returns
    -------

    """
    # load a Sdata using its Id
    filename = getloc(date, index, mindex, rootdir=rootdir)
    print('Sdata_manip.load_Sdata(): filename = ' + filename)
    S = load_Sdata_file(filename)
    return S


def read(filename, data=False):
    S = pickle_m.read(filename)

    if data:
        S.load_measure()
    return S


def getloc(date, index, mindex, frmt='.hdf5', rootdir=None):
    """Return the filename associated with the cine or hdf5 file in supplied datadir or in a directory with the
    supplied date that is in the current $PATH"""
    datadir = getdir(date, rootdir=rootdir)
    filename = datadir + "Sdata_" + date + "_" + str(index) + "_" + str(mindex) + frmt
    print 'Sdata_manip.getloc(): filename = ', filename
    return filename


def getloc_S(S, frmt='.hdf5'):
    Dir = getdir(S.id.date)
    filename = Dir + "Sdata_" + S.id.date + "_" + str(S.id.index) + "_" + str(S.id.mindex) + frmt
    return filename


def getdir(date, rootdir=None):
    if rootdir is None:
        rootdir = file_architecture.get_dir(date)
    Dir = os.path.join(rootdir, "Sdata_" + date + "/")
    return Dir


def main():
    # date='2015_03_24'
    Sdata_gen_day(date)
    Measure_gen_day(date)

    # main()
