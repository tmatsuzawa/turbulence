# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:43:49 2015

@author: stephane, takumi
"""

import turbulence.cine as cine
from scipy import misc
import os
import numpy as np
import turbulence.tools.browse as browse
import turbulence.tools.rw_data as rw_data
# Convert cine file to a list of .TIFF images
import glob

import sys
# sys.path.append('/Users/stephane/Documents/git/takumi/turbulence')

def cine2tiff(file,mode,step,start=0,stop=0,ctime=1,folder='/Tiff_folder',post=''):
    """
    Generate a list of tiff files extracted from a cinefile.
        Different modes of processing can be used, that are typically useful for PIV processings :
        test : log samples the i;ages, using the function test_sample. Default is 10 intervals log spaced, every 500 images.
        Sample : standard extraction from start to stop, every step images, with an interval ctime between images A and B.
        File : read directly the start, stop and ctime from a external file. Read automatically if the .txt file is in format :
        'PIV_timestep'+cine_basename+.'txt'
    INPUT
    -----
    file : str
        filename of the cine file
    mode : str. 
        Can be either 'test','Sample', 'File'
        single : list of images specified
        pair : pair of images, separated by a ctime interval
    step : int
        interval between two successive images to processed.
    start : int. default 0
        starting index
    stop : int. default 0. 
        The cine will be processed 'till its end
    ctime : 
    folder : str. Default '/Tiff_folder'
        Name of the root folder where the images will be saved.
    post : str. Default ''
        post string to add to the title of the tiff folder name
    OUTPUT
    OUTPUT
    None
    """
    #file : path of the cine file
    print mode
    try:
        c = cine.Cine(file)     
    except:
        print('Cine file temporary unavailable')
        return None
        
    print('cine open')
    print('Length : '+str(len(c)))
    
    # get maximum value according to bit depth
 #   bitmax = float(2**c.real_bpp)
    #indexList=sampleCine(c,mode,step,start,stop,ctime)
    if mode=='test':
        post='test_logSample'
        indexList,ndigit=test_sample(c,start,stop)

    if mode=='Sample':
        #Use that mode : give the 
      #  indexList,ndigit=manual_sampling([1000],[3000],[1],10)
        indexList,ndigit=manual_sampling([start],[stop],Dt=[step],step=ctime)
        print(indexList)
    if mode=='File':       
        filename=os.path.dirname(file)+'/PIV_timestep_'+os.path.basename(file)[:-5]+'.txt'
       # print(filename)
        indexList, ndigit=sample_from_file(filename,step=step)
        #print(indexList)

    savefile = os.path.dirname(file)+folder+'/'+os.path.basename(file)
    Dir,root=saveDir(savefile, post+'')
  #  Dir='/Users/stephane/Documents/Experiences_local/Accelerated_grid/2015_03_21/PIV_bv_hp_Zm80mm_fps10000_H1000mm_zoom_S100mm/'+post+'/'
    if not os.path.isdir(Dir):
        os.makedirs(Dir)

    print(indexList)
    for index in indexList:
        # get frames from cine file
        filename=Dir+root+browse.digit_to_s(index,ndigit)+'.tiff'
        #save only if the image does not already exist !
        if not os.path.exists(filename):
            print(filename, index)
            
            if index<len(c):
                data = c.get_frame(index)            
                misc.imsave(filename,data,'tiff')
#        im=Image.fromarray(data)
#        im.save(filename,'tiff')
#        a = a / bitmax * 255        
#        im_ref = np.float64(a)
    c.close()
    
def test_sample(c,start=0,stop=0):
    """
    Generate a list of index to process. stacks of 10 logspace indexes, regurlarly spaced.
    INPUT
    -----
    c : Cine object.
        Only use to recover the number of images (?). 
        should it be replaced directly by the number of images ?
    start : int. Default 0
        starting index
    stop : int. Default 0 (processed 'till the end)
        stop index
    OUTPUT
    ----- 
    """

    N=len(c.image_locations)
    print(N)
  #sample logarithmically a file, every 1000 images, to look for the rigth time scale of evolution
  #  "adaptative PIV parameters"   
    if (stop==0) or (stop>N):
        stop=N #for single !!!
    
    Dt=[0,1,2,3,5,10,20,50,100,200]
    step=500        

    #for logmovie !!!!!!!!!!
    Dt=[0,1,2,3,4,5,10,15,20,50]
    step=200        

    K=np.cumsum(Dt)
    
    indexList=[]    
    for i in range(start,stop-max(Dt),step):
        index=[i+k for k in K]
        indexList=indexList+index
#        indexName=indexName+nam     e   
    
    ndigit=len(str(N))
    return indexList,ndigit
    
def manual_sampling(start,end,Dt=[1],step=1, piv=True):
    """
    Returns a list of indices which indicates when to save tiff files from a cine.
    i.e. [0,1, 10, 11] will be passed so that cine2tiff will save images at only those frames.
    Parameters
    ----------
    start
    end
    Dt
    step
    piv

    Returns
    -------

    """
    #start : start image, list format 
    #end :  end image list format
    #Dt : time step between images A and B for PIV analysis. list format
    #step : time step (in multiple of Dt) between two successive pairs of images A/B

    #debut=[1000]#,1200,1700,2200,4500,8500,17500]
    #fin=[3000]#,1700,2200,4500,8500,17500,25000]
    #Dt=[1 2 3 5 10 20 50]
    ndigit=int(max(round(np.log10(end[-1])),5))
    #step=10
    
    indexList=[]
    for i in range(len(start)):
        d=start[i]
        f=end[i]
        pas=Dt[i]

#       index=[d+j for j in np.arange(0,f-d,pas)]     
        indexA=[d+j for j in np.arange(0,f-d-pas,step)]
        if piv:
            indexB=[d+j for j in np.arange(pas,f-d,step)]
        else:
            indexB=[]

        #store the new indexes in a list
        indexList=indexList+indexA+indexB
    indexList.sort()
    print('number of images to save : '+str(len(indexList)))
    
    return indexList, ndigit

def sample_from_file(filename,step=1):
    #file must contain a list of starting and ending index, associated with a time step Dt
    # filename : name of the parameter file
    # step : optional time step between two successive pairs of images. If step=1, the step will be equal to Dt
    Header,data=rw_data.read_dataFile(filename,'\t','\t')

    start=[int(p) for p in data['start']]
    end=[int(p) for p in data['end']]
    Dt=[int(p) for p in data['Dt']]

    return manual_sampling(start,end,Dt,step)
        
def sampleCine(c,mode,step,start=0,stop=0,ctime=1):
    N=len(c.image_locations)
    
    if (stop==0) or (stop>N):
        stop==N #for single !!!

    #Mode : pair, single, series
    #for series, step become a List (!) of step
    if mode=='single':
        indexList=range(start,stop,step)
    if mode=='pair':
        indexL1=[i for i in range(start,stop-ctime,step)]
        indexL2=[i for i in range(start+ctime,stop,step)]
        indexList=indexL1+indexL2
        indexList.sort()
    if mode=='series':
        indexList=[]
        for dt in step:
            index=range(start,stop,dt)
            indexList=indexList+index
        indexList.sort()
        
    return indexList
    
def saveDir(file,post=''):
    fileroot=browse.get_string(file,'','.cine', 0)
    if not (fileroot==''):
        Dir=fileroot+post+'/'
        root='im'
        if not os.path.isdir(Dir):
            os.makedirs(Dir)
    else:
        print('not a cine file')
        
    return Dir,root


def cine2tiff_for_pivstacks(file, mode, step, start=0, stop=0, ctime=1, offset=0, folder='/Tiff_folder', post=''):
    """

    INPUT
    -----
    file : str
        filename of the cine file
    mode : str.
        Can be either 'test','Sample', 'File'
        single : list of images specified
        pair : pair of images, separated by a ctime interval
    step : int
        interval between two successive images to processed.
    start : int. default 0
        starting index
    stop : int. default 0.
        The cine will be processed 'till its end
    ctime :
    folder : str. Default '/Tiff_folder'
        Name of the root folder where the images will be saved.
    post : str. Default ''
        post string to add to the title of the tiff folder name
    OUTPUT
    OUTPUT
    None
    """
    # file : path of the cine file


    try:
        c = cine.Cine(file)
    except:
        print('Cine file temporary unavailable')
        return None

    print('cine open')
    print('Length : ' + str(len(c)))

    # get maximum value according to bit depth
    #   bitmax = float(2**c.real_bpp)
    # indexList=sampleCine(c,mode,step,start,stop,ctime)
    if mode == 'test':
        post = 'test_logSample'
        indexList, ndigit = test_sample(c, start, stop)

    if mode == 'Sample':
        # Use that mode : give the
        #  indexList,ndigit=manual_sampling([1000],[3000],[1],10)
        indexList, ndigit = manual_sampling([start], [stop], Dt=[step], step=ctime)
        print(indexList)
    if mode == 'File':
        filename = os.path.dirname(file) + '/PIV_timestep_' + os.path.basename(file)[:-5] + '.txt'
        # print(filename)
        indexList, ndigit = sample_from_file(filename, step=step)
        # print(indexList)

    savefile = os.path.dirname(file) + folder + '/' + os.path.basename(file)
    Dir, root = saveDir(savefile, post + '')

    if not os.path.isdir(Dir):
        os.makedirs(Dir)

    for index in indexList:
        # get frames from cine file
        filename = Dir + root + browse.digit_to_s(index, ndigit) + '.tiff'
        # save only if the image does not already exist !
        if not os.path.exists(filename):
            print(filename, index)

            # # the following conditions were used for 20171127 data (2pt, 40pt per cycle, duty cycle=85%)
            # if index < len(c):
            #     if (35 > index % 40 > 0):
            #         data = c.get_frame(index)
            #         misc.imsave(filename, data, 'tiff')

            if index < len(c):
                a = index % 60
                if (a % 3 != offset) and (a < 51):
                    data = c.get_frame(index)
                    misc.imsave(filename, data, 'tiff')




    c.close()

