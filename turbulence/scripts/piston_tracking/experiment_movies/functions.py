import cine
from pylab import *
from scipy import ndimage
from scipy.optimize import curve_fit
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
import os
import PIL.ImageOps
import PIL.Image
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import cPickle as pickle
import math
import itertools
import time
import pylab as P
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
import sys


def load_pickled_data(root_dir, filename = -1):
    '''loads data from pickle file .

    Parameters
    ----------
    root_dir : string
        Directory in which file is saved
    filename : string 
        name of file

    Returns
    ----------
    data : any python object
        data from pickled file
        '''
    if filename ==-1:
        tot_path = root_dir
    else:
        tot_path = root_dir + '/' + filename

    try :
        of = open(tot_path, 'rb')
        data = pickle.load(of)
    except Exception:
        data = 0
        print 'file not found', tot_path
        sys.exit()
    return data

def dump_text_data(output_dir, filename, data):
    con = open(output_dir + '/' + filename + '.csv', "wb")
    con_len = len(data)
    data = array(data)
    print data[0]
    print data[con_len-1]

    for i in range(con_len):
        for j in range(len(data[0])):
            #print i, j
            #print data[i,j]
            con.write(str(data[i,j]) + ' ,') 
        con.write('\n')
    con.close()

def dump_pickled_data(output_dir, filename, data):
    if not os.path.exists(output_dir): os.mkdir(output_dir)
    of = open(output_dir + '/'+filename + '.pickle', 'wb')
    pickle.dump(data, of, pickle.HIGHEST_PROTOCOL)
    of.close()
    
def func(x, xmax, tau, f, phi):
    return (xmax*exp(-1/tau * x)*sin(2*pi* f *x + phi))
   
   
def shift(yvals):
    y_mean = mean(yvals)
    yvals = yvals - y_mean
    return yvals

def pick_point(fn, ff, c):
    minval, maxval = 15, 60#60, 2000
    frame = (clip(c[ff].astype('f'), minval, maxval)-minval)/(maxval-minval)
    #frame = c[0].astype('f')/2**c.real_bpp
    m_bf = mean(frame)
    std_bf = std(frame)
        
    minval = m_bf -10.*std_bf
    maxval = m_bf + 500.*std_bf

    f = (clip(c[ff].astype('f'), minval, maxval)-minval)/(maxval-minval)
    cine.asimage(f).save(fn + '.png')

    fig = plt.figure()
    img =  mpimg.imread(fn + '.png')
    imgplot = plt.imshow(img, cmap= cm.Greys_r)


    fig.canvas.mpl_connect('button_press_event', on_key)
    plt.show()
    N = mutable_object['key']
    #print array(N)[0] 
    
    return array(N)

def find_track(f, frame, output_dir, pix = 2):
    
    frame_mean = mean(f.flatten())
    
    cine.asimage(f).save(output_dir + 'convolved.png')

    
    data_max = filters.maximum_filter(f, 70)
    maxima = (f == data_max)
    
    data_min = filters.minimum_filter(f,70)

    
    dmax = max((data_max-data_min).flatten())
    dmin = min((data_max-data_min).flatten())
    
    minmax = (dmax-dmin)
    
    
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #n, bins, patches = ax.hist((data_max-data_min).flatten(), 100, normed=1, facecolor='green', alpha=0.75)
    #plt.show()   


    diff = ((data_max - data_min) >= dmin + 0.46*minmax)
    maxima[diff == 0] = 0 
    
    
    labeled, num_object = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    
    x,y = [], []
    print len(slices)
    for dy, dx in slices:
       
        rad = sqrt((dx.stop-dx.start)**2 +(dy.stop-dy.start)**2)
        #print 'rad', rad
        
        if rad < 25 and rad >1.0:
            #print ra
            x_center = (dx.start+dx.stop)/2
            x.append(x_center)
            y_center = (dy.start + dy.stop)/2
            y.append(y_center)

    fig = plt.figure()
    img =  mpimg.imread(output_dir + 'original.png')
    imgplot = plt.imshow(img, cmap= cm.Greys_r)
    plt.plot(x,y, 'ro')
    
    
    
    
    #cine.asimage(f).save(fn + '.png')


    #fig.canvas.mpl_connect('button_press_event', on_key)
    #plt.show()
    #N = mutable_object['key']
    #print array(N)[0]

    
    points = array([x,y]).T
    for i in range(len(points)):
        trackpoint = points[i]
        for j in range(4):
            
            bf = frame[trackpoint[1]-pix:trackpoint[1]+pix]
            bf = bf[:, trackpoint[0]-pix:trackpoint[0]+pix]
    
            com = ndimage.measurements.center_of_mass(bf.astype('f'))
 
            t2 = ones_like(bf, dtype = 'f')
            ce = array(ndimage.measurements.center_of_mass(t2.astype(float)))
            

            
            movx = ce[0] - com[0]#pix - com[0]
            movy = ce[1] - com[1]#pix - com[1]
            
            if math.isnan(movx):
                movx = 0
            if math.isnan(movy):
                movy =0
    

            points[i,0] = trackpoint[0] - movy
            points[i,1] = trackpoint[1] - movx
    
    
    return points

def track_points(f, points, pix, f_num, l_num, output_dir):
    num_points = len(points)
    points = points.astype(float)
    gamma = 2.2

    frames = []
    frames_last = []
    
    for i in range(num_points):
        
        try:
            trackpoint = points[i].astype('f')
           
            
            bf = f[trackpoint[1]-pix:trackpoint[1]+pix]
            bf = bf[:, trackpoint[0]-pix:trackpoint[0]+pix]
            
            sh_bf = array([2*pix, 2*pix])
            
            m_bf = mean(bf)
            std_bf = std(bf)
            #maxval = 0.83*max(f.flatten())
            #minval = 0.82*max(f.flatten())
            
            
            #f = (clip(c[i+ff].astype('f'), minval, maxval)-minval)/(maxval-minval)
            
            
            
            if f_num ==0:
                if array_equal(shape(bf), sh_bf):
                    frames.append((bf**(1./gamma) * 255).astype('u1'))
                if i == (num_points-1):
                    cine.asimage(hstack(frames)).save(output_dir + 'initial_points' + '.png')
        
        
            for j in range(2):
                bf = f[trackpoint[1]-pix:trackpoint[1]+pix]
                bf = bf[:, trackpoint[0]-pix:trackpoint[0]+pix]
        
                com = ndimage.measurements.center_of_mass(bf.astype('f'))
            
                
            
                t2 = ones_like(bf, dtype = 'f')
                ce = array(ndimage.measurements.center_of_mass(t2.astype(float)))
                            
                
                
                movx = float(ce[0]) - float(com[0])#pix - com[0]
                movy = float(ce[1]) - float(com[1])#pix - com[1]
        
                if math.isnan(movx):
                    movx = 0
                if math.isnan(movy):
                    movy =0
                    

                
                points[i,0] = float(trackpoint[0]) - float(movy)
                points[i,1] = float(trackpoint[1]) - float(movx)

            if f_num ==(l_num-1):
                if array_equal(shape(bf), sh_bf):
                    frames_last.append((bf**(1./gamma) * 255).astype('u1'))
                if i == (num_points-1):
                    cine.asimage(hstack(frames_last)).save(output_dir + 'final_points' + '.png')
                
           
        except RuntimeError:
            points[i] = points[i]
            print 'runtime error'
    
    return points
    
   
mutable_object = {}
def on_key(event):
    N=[event.ydata, event.xdata]
    mutable_object['key'] = N
    plt.close()
    #print N

def find_files_by_extension(root_dir, ext, tot=False):
    filenames  = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(ext):
                if tot == False:
                    filenames.append(file)
                else :
                    filenames.append(root+'/' + file)
            #if file.endswith(ext):
               # filenames.append(root + '/' + file)
    return filenames
                


def find_files_by_name(root_dir, name, tot):
    filenames  = []
    for root, dirs, files in os.walk(root_dir):
        
        if name in files:
            if tot:
                filenames.append(root + '/' + name)
            else :
                filenames.append(root)
                
    return filenames

def my_fft(f, ts, omegas):
    fft_res = []
    len_omegas = len(omegas)
    
    for i in range(len_omegas):
        omega = omegas[i]
        ff = f*exp(2*1j*pi*omega*ts)
        fft_res.append(sum(ff))
    
    return array(fft_res)

def isplit(iterable,splitters):
    return [list(g) for k,g in itertools.groupby(iterable,lambda x:x in splitters) if not k]

def track_and_save(fn, vid_dirs, img_ker_string):
    'tracking...'
    start_time = time.time()
    img_ker =  mpimg.imread(img_ker_string)

    c = cine.Cine(fn)
    tot_frames = len(c)
    com_data = []
    gamma = 2.2
    
    ff= 0
    lf = tot_frames
    lf = int(lf)
    ff = int(ff)

    rd_s = '/Volumes/labshared3/Lisa'


    output_dir_i = fn.split('.')[0]
    output_dir_i2 = output_dir_i.split('/')
    

    output_dir = rd_s + '/' + time.strftime("%Y_%m_%d") +'_video_tracks_' 
    if not os.path.exists(output_dir):os.mkdir(output_dir)
        
    output_dir = rd_s + '/' + time.strftime("%Y_%m_%d") +'_video_tracks_' + '/'+vid_dirs+ '_' +output_dir_i2[-1]+'/'
    if not os.path.exists(output_dir):os.mkdir(output_dir)    
    
    com_data = []
    
    tot_frames = lf
    val = 0.8 * max(c[ff].astype(float).flatten())
    
    points = zeros([tot_frames-ff, 2], dtype = 'f')
    for i in range(tot_frames-ff):
        

        
        if i ==0 :
            minval = .04*val
            maxval = 0.13*val
            
        else:
            minval = 0.04*val
            maxval = 0.13*val

        frame = (clip(c[ff+i].astype('f'), minval, maxval)-minval)/(maxval-minval) 
        
        if i == 0:
            cine.asimage(frame).save(output_dir + 'original.png')
            print 'saved original', output_dir
        
            fr = ndimage.convolve(frame, img_ker, mode = 'reflect', cval = 0.0) 
            minval = 0.*max(fr.flatten())
            maxval = 1*max(fr.flatten())
            fr = (clip(fr, minval, maxval)-minval)/(maxval-minval)

            points = find_track(fr, frame,output_dir)
        
            pix = 7
            
            fig = plt.figure()
            img =  Image.open(output_dir + 'original.png')
            img = array(img)
            
            for j in range(len(points)):    
                img[points[j,1]-pix: points[j,1]+pix, points[j,0]-pix: points[j,0]+pix] = array(ImageOps.invert(Image.fromarray(np.uint8(img[points[j,1]-pix: points[j,1]+pix, points[j,0]-pix: points[j,0]+pix]))))
    
            plt.imshow(img,cmap= cm.Greys_r)
            plt.savefig(output_dir + 'tracked_areas.png')
            

        points = track_points(frame.astype('f'), points, pix, i, tot_frames-ff, output_dir).astype(float)
          
        
        t = array([[ff+i, points[j,0], points[j,1]] for j in range(len(points))])            
        com_data.append(t)

    

    com_data = array(com_data)
    sh1, sh2, sh3 = shape(com_data)
    com_data = com_data.flatten()
    com_data = reshape(com_data, [sh1*sh2,3])
    
    num_times = len(com_data[::,0])
    
    for i in range(num_times):
        com_data[i,0] = c.get_time(int(com_data[i,0]))

    dump_pickled_data(output_dir, 'com_data', com_data)
    
    c.close()
    
    end_time = time.time()
    
    total_time =  end_time-start_time
    
    print 'tracked ',  tot_frames, 'frames in' , total_time , 'seconds'
    
    return com_data, output_dir

def color_by_speed(fn, root_dir):
    dat = load_pickled_data(fn)
    speeds = load_pickled_data(root_dir+'speed.pickle')

    
    t_dat = (dat.T)[0] 
    num_gyros =  len(t_dat[where(t_dat==0)])

    
    x_dat = (dat.T)[1]
    y_dat = (dat.T)[2]
    
    
    num_time_steps = len(x_dat)/num_gyros
    fig = plt.figure()
    img =  mpimg.imread(root_dir + 'original.png')
    imgplot = plt.imshow(img, cmap= cm.Greys_r)
    
    labels = ['{0}'.format(k) for k in range(num_gyros)]

    x_gy = []
    y_gy = []
    rad = []
    patch = []
    #print len(num_gyros)
    print len(speeds)
    for j in range(len(speeds)):
        index = int(speeds[j,0])

        x_gy.append((x_dat[index]))
        y_gy.append((y_dat[index]))
        rad.append(30)
        
        circ = Circle((x_dat[index] ,y_dat[index]), 30)
        patch.append(circ)
    speeds = array(speeds).T[1] 
    
    mean_speeds = mean(abs(speeds))
    std_speeds = std(abs(speeds))
    
    disps =  (abs(speeds)-mean_speeds)/(mean_speeds)
    
    min_lim = 250#mean_speeds - 2*std_speeds
    max_lim = 350#mean_speeds + 2*std_speeds
    print 'min lim', speeds
    p_ax = P.axes()
    p = PatchCollection(patch, cmap = 'bwr', alpha = 0.5)
    p.set_array(P.array(abs(speeds)))
    p.set_clim([min_lim, max_lim])
    p_ax.add_collection(p)
    plt.colorbar(p)
    #p_ax.axes.get_xaxis().set_visible(False)
    #p_ax.axes.get_yaxis().set_visible(False)
    p_ax.axes.get_xaxis().set_ticks([])
    p_ax.axes.get_yaxis().set_ticks([])
    
    plt.savefig(root_dir + 'color_by_speed_new.png')
    dump_pickled_data(root_dir, 'disps.pickle', disps)
   
   
def moving_average(values, window):
    weigths = repeat(1.0, window)/window
    smas = convolve(values, weigths, 'valid')
    return smas
    
def check_time(time_s1):
    '''
    Checks to see if it has been long enough since file creation to do tracking.
    
  Parameters
        -----------------
       time_s1: string
            
    
        Returns
        ---------------
        ok_to_continue: boolean
            tells you if it has been long enough since file creation to continue
             
    
    
    '''
    c_time = time.ctime()
    
    #compare years first
    c_time = c_time.split(' ')
    time_s1 = time_s1.split(' ')
    
    c_time_stamp = c_time[3]
    time_s1_stamp = time_s1[3]
    
    c_time[3] = 0
    time_s1 [3] = 0
    
    print c_time
    print time_s1
    
    if c_time != time_s1:
        ok_to_continue = True
    else:
        split_c = c_time_stamp.split(':')
        split_t = time_s1_stamp.split(':')
        
        print split_c
        print split_t
        
        #compare hour
        if split_c[0] != split_t[0]:
            ok_to_continue = True
        else:
            min_c = int(split_c[1])
            min_t = int(split_t[1])
            
            print min_c
            print min_t
            
            time_between = abs(min_c - min_t) #time since creation of file in minutes
            print time_between
            if time_between < 3:
                ok_to_continue = False
            else:
                ok_to_continue = True
    return ok_to_continue

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
   
        

