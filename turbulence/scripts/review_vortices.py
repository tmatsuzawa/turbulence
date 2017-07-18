

import numpy as np

import glob
import os
import stephane.tools.browse as browse

import stephane.tools.rw_data as rw_data
import stephane.manager.file_architecture as file_architecture

import stephane.mdata.Sdata_manip as Sdata_manip



def review():
    folder = '/Volumes/Stephane/Vortex_Turbulence/'

    folders = glob.glob(folder+'*')

    l_vortices = []

    D = {}
    D['v']=[]
    D['fx']=[]
    D['fps']=[]
    D['date']=[]
    D['index']=[]
#D['Sdata_file']=[]
    D['cinefile']=[]


    for f in folders:
    
    #print(f)
        l = glob.glob(f+'/PIV*A0mm*.cine')
        l_vortices = l_vortices+l
       
#print(l_vortices)

    for cinefile in l_vortices:
        s = browse.get_string(cinefile,'','/PIV')
        date = s[-10:]
    
        filename = os.path.dirname(file_architecture.os_i(cinefile))+'/Sdata_'+date+'/Cine_index_'+date+'.txt'
        if os.path.exists(filename):
        
            Header,data = rw_data.read_dataFile(filename,Hdelimiter='\t',Ddelimiter='\t')  
      #  print(data)  
       # print(data)
        #print("l : "+l)
    #    print(data['Names'])
       # print(file_architecture.os_i(cinefile))
            index = np.where([file_architecture.os_i(cinefile)==file_architecture.os_i(name) for name in data['Names']])[0][0]
            index = int(data['Index'][index])
        
            v = browse.get_number(cinefile,'_v','mms')
            if v==-1:
                v = browse.get_number(cinefile,'mm_v','mm')
            fps = browse.get_number(cinefile,'_fps','_')

            S = Sdata_manip.load_Sdata(date,index)
            fx = S.param.fx
        
            D['date'].append(date)
            D['index'].append(index)
            D['v'].append(v)
            D['fps'].append(fps)
            D['fx'].append(fx)
#        D['Sdata_file'].append(filename)
        
            D['cinefile'].append(cinefile)
        

    print(D['v'])

    filename = './Vortices/free_index_no_cine.txt'
    rw_data.write_a_dict(filename,D,delimiter='\t')

    
def dispersion_trajectories(date,indices):
    savedir = './Vortex_Turbulence/Overview/'+date+'/'
    
    Slist = Sdata_manip.load_serie(date,indices)
    Mlist = Sdata_manip.load_measures(Slist,indices=0)
    
    for M in Mlist:
        M.get('E')
        M.get('omega')
    
    for M in Mlist:
        if M.Id=='pointer':
            M.Id = M.Sdata.Id
        
    pos = []
    for i,M in enumerate(Mlist):
        pos.append(track.position(M,sigma=2.))
    
    tmin = 20
    tmax = 158

    accurate=[]
    for i,M in enumerate(Mlist):
        figs,acc = track.plot(pos[i],tmin,tmax)
        accurate.append(acc)
    graphes.save_figs(figs,savedir=savedir+'')
    
    
"""
for l in l_vortices:

        
    dt = browse.get_number(l,'_fps','_')
    
    #print(v,dt)
    

#print(l_vortices)
#print(len(l_vortices))
"""
def main():
    dispersion_trajectories(args.date,args.indices)
    
if __name__ == '__main__':
    main()
