function process_dir(Dirbase)
%Dirbase = '/Volumes/labshared3-1/takumi/2018_02_01';
Dir = [Dirbase '/Tiff_folder'];
cd(Dir)

Dirs = dir('PIV_*');
dirlist = {Dirs.name};
disp(dirlist')

Dt = 1; % process PIV from image n and image n+Dt
step = 2; %process image n & imange n+Dt pair, then n+step & (n+step)+Dt
W = 32; %width in pixel of the box to compute velocity 

treatpiv_t(Dirbase,dirlist,Dt,W,step,false)