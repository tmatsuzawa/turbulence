import h5py
import numpy as np
import glob
import turbulence.tools.rw_data as rw_data
import turbulence.tools.browse as browse
import turbulence.jhtd.cutout as cutout
import datetime
import turbulence.jhtd.get as jhtd_get
import matplotlib.pyplot as plt
import turbulence.display.graph as graph

"""
Modules to analyze JHTD data
author: takumi
"""

## Simulation parameters
# Grid: 1024^3
# Domain: 2pi x 2pi x 2pi  <- range of x,y,z is [0,2pi] each.
dx = 2 * np.pi / 1024.
dy = dz = dx
dt = 0.002  # time separation between data points (10 DNS steps)
# dt_sim = 0.0002  #simulation timestep
nu = 0.000185  # viscosity



def read_v_on_xy_plane_at_t0(filepath,z=0):
    """
    Generate 2d numpy arrays for Ux(z=0, y, x,t=0) and Uy(z=0, y, x,t=0) using 1 x Ny x Nx x t h5 data for isotropic turbulence DNS simulation from JHTD.

    INPUT
    -----
    filepath : str
        definite path to the h5 file from JHTD
    z: int
        z values that define the slice

    OUTPUT
    ------
    ux0: 2d numpy array
        ux=ux(y,x) at t=t0
    uy0: 3d numpy array
        uy=uy(y,x) at t=t0
    uz0: 3d numpy array
        uz=uz(y,x) at t=t0
    """

    # Read parameters from the filename
    param = jhtd_get.get_parameters(filepath)

    # Read velocity field at t=t0
    with h5py.File(filepath, 'r') as f:
        key = 'u0000{0}'.format(param["t0"])
        u = f[key][()] ## u is 3-dim array. (xl x yl x zl) Elements of u are also 3 dim arrays.
        f.close()

    # Obtain ux,uy,uz fields
    ux = u[:, :, :, 0]  #ux(z,y,x)
    uy = u[:, :, :, 1]
    uz = u[:, :, :, 2]

    # Obtain the velocity field on the z=0 plane
    ux0 = ux[z, :, :]  # ux(z=0,y,x)
    uy0 = uy[z, :, :]  # ux(z=0,y,x)
    uz0 = uz[z, :, :]  # uz(z=0,y,x)

    print 'Extracted the velocity field on the z=0 plane! (t=t0)'
    return ux0, uy0, uz0

def generate_XYZT_arrays(param):
    """
    Generate 1d arrays of time and positions. The main use is to plot graphs.
    Parameters
    ----------
    param: dict, this stores values used to get a data cutout from the JHTD
        This can be obtained from the output of jhtd_get.get_parameters(filepath)
    Returns
    -------
    Returns
    -------
    T,X,Y,Z: 1d array (float32)
    """
    T = [param['t0'] + dt * i for i in range(param['tl'])]
    X = [param['x0'] + dx * i for i in range(param['xl'])]
    Y = [param['y0'] + dy * i for i in range(param['yl'])]
    Z = [param['z0'] + dz * i for i in range(param['zl'])]
    return T, X, Y, Z


def compute_lambda_local(filepath, savedir, save=True):
    """
    Computes LOCAL Taylor microscale (lambda) and LOCAL Re_lambda using Ux and Uy

    INPUT
    -----
    filepath: str, definite path to the h5 file from JHTD
    savedir: str,  definite path to the h5 file from JHTD
    OUTPUT
    ------


    """
    # Read parameters from the filename
    param = jhtd_get.get_parameters(filepath)


    ux0, uy0, uz0 = read_v_on_xy_plane_at_t0(filepath, z=0)

    # Convert 2d numpy arrays (Ny,Nx) to 3d numpy arrays (Ny,Nx,1)
    Ux=np.reshape(ux0,ux0.shape+(1,))
    Uy=np.reshape(uy0,uy0.shape+(1,))
    Uz=np.reshape(uz0,uz0.shape+(1,))

    Ux2 = np.zeros((param["yl"], param["xl"], param["tl"]))
    Uy2 = np.zeros((param["yl"], param["xl"], param["tl"]))
    U2 = np.zeros((param["yl"], param["xl"], param["tl"]))
    dUxdx = np.zeros((param["yl"], param["xl"], param["tl"]))
    dUydy = np.zeros((param["yl"], param["xl"], param["tl"]))
    dUxdx2 = np.zeros((param["yl"], param["xl"], param["tl"]))
    dUydy2 = np.zeros((param["yl"], param["xl"], param["tl"]))
    dUidxi2 = np.zeros((param["yl"], param["xl"], param["tl"]))
    lambdaT_Local = np.zeros((param["yl"], param["xl"], param["tl"]))
    Re_lambdaT_Local = np.zeros((param["yl"], param["xl"], param["tl"]))
    lambdaTx_Local = np.zeros((param["yl"], param["xl"], param["tl"]))
    Re_lambdaTx_Local = np.zeros((param["yl"], param["xl"], param["tl"]))
    lambdaTy_Local = np.zeros((param["yl"], param["xl"], param["tl"]))
    Re_lambdaTy_Local = np.zeros((param["yl"], param["xl"], param["tl"]))

    Dx = Dy = dx

    print 'Computing Local Taylor microscale (lambda) using Ux and Uy...'
    # Compute spatial gradient of Ux and Uy
    for t in range(0, param["tl"]):
        for x in range(0, param["xl"] - 1):
            for y in range(0, param["yl"] - 1):
                Ux2[y, x, t] = Ux[y, x, t] * Ux[y, x, t]
                Uy2[y, x, t] = Uy[y, x, t] * Uy[y, x, t]
                U2[y, x, t] = Ux2[y, x, t] + Uy2[y, x, t]
                if x == param["xl"] - 1 or y == param["yl"] - 1:
                    continue
                dUxdx[y, x, t] = (Ux[y, x + 1, t] - Ux[y, x, t]) / Dx  # du'_x/dx
                dUydy[y, x, t] = (Uy[y + 1, x, t] - Uy[y, x, t]) / Dy  # du'_y/dy

                dUxdx2[y, x, t] = dUxdx[y, x, t] * dUxdx[y, x, t]
                dUydy2[y, x, t] = dUydy[y, x, t] * dUydy[y, x, t]
                dUidxi2[y, x, t] = dUxdx2[y, x, t] + dUydy2[y, x, t]

    # Time average of U^2
    Ux2ave = np.nanmean(Ux2, axis=2)
    Uy2ave = np.nanmean(Uy2, axis=2)
    U2ave = np.nanmean(U2, axis=2)
    dUxdx2ave = np.nanmean(dUxdx2, axis=2)
    dUydy2ave = np.nanmean(dUydy2, axis=2)
    dUidxi2ave = np.nanmean(dUidxi2, axis=2)

    # Calculate LOCAL Taylor microscale:
    dUxdxCutoff = 10**(-3)  #
    for x in range(0, param["xl"] - 1):
        for y in range(0, param["yl"] - 1):
            if dUidxi2ave[y, x] < dUxdxCutoff:
                lambdaT_Local[y, x] = 0
            else:
                lambdaT_Local[y, x] = np.sqrt(U2ave[y, x] / dUidxi2ave[y, x])
                Re_lambdaT_Local[y, x] = lambdaT_Local[y, x] * np.sqrt(1.0 / 2.0 * U2ave[y, x]) / nu  # b/c Ux2 ~ 1/2 U2
            if dUxdx2ave[y,x]< dUxdxCutoff:
                lambdaTx_Local[y,x]=0
            else:
                lambdaTx_Local[y,x] = np.sqrt(Ux2ave[y,x]/dUxdx2ave[y,x])
                Re_lambdaTx_Local[y,x] = lambdaTx_Local[y,x]*np.sqrt(Ux2ave[y,x])/nu
            if dUydy2ave[y,x]< dUxdxCutoff:
                lambdaTy_Local[y,x]=0
            else:
                lambdaTy_Local[y,x] = np.sqrt(Uy2ave[y,x]/dUydy2ave[y,x])
                Re_lambdaTy_Local[y,x] = lambdaTy_Local[y,x]*np.sqrt(Uy2ave[y,x])/nu

    # Print and Savespatial averages of Local Re_lambda and Local lambda
    if save:
        save_computed_results(U2ave, Ux2ave, Uy2ave, dUidxi2ave, dUxdx2ave, dUydy2ave, savedir, save)
        save_computed_results_Local(lambdaT_Local, Re_lambdaT_Local, lambdaTx_Local, Re_lambdaTx_Local, lambdaTy_Local, Re_lambdaTy_Local, savedir)

    return lambdaT_Local, Re_lambdaT_Local, lambdaTx_Local, Re_lambdaTx_Local, lambdaTy_Local, Re_lambdaTy_Local

def compute_lambda(filepath, savedir, save=True):
    # Read parameters from the filename
    param = jhtd_get.get_parameters(filepath)

    ux0, uy0, uz0 = read_v_on_xy_plane_at_t0(filepath, z=0)

    # Convert 2d numpy arrays (Ny,Nx) to 3d numpy arrays (Ny,Nx,1)
    Ux = np.reshape(ux0, ux0.shape + (1,))
    Uy = np.reshape(uy0, uy0.shape + (1,))
    Uz = np.reshape(uz0, uz0.shape + (1,))

    Ux2 = np.zeros((param["yl"], param["xl"], param["tl"]))
    Uy2 = np.zeros((param["yl"], param["xl"], param["tl"]))
    U2 = np.zeros((param["yl"], param["xl"], param["tl"]))
    dUxdx = np.zeros((param["yl"], param["xl"], param["tl"]))
    dUydy = np.zeros((param["yl"], param["xl"], param["tl"]))
    dUxdx2 = np.zeros((param["yl"], param["xl"], param["tl"]))
    dUydy2 = np.zeros((param["yl"], param["xl"], param["tl"]))
    dUidxi2 = np.zeros((param["yl"], param["xl"], param["tl"]))

    Dx = Dy = dx

    print 'Computing Taylor microscale (lambda) using Ux and Uy...'
    # Compute spatial gradient of Ux and Uy
    for t in range(0, param["tl"]):
        for x in range(0, param["xl"] - 1):
            for y in range(0, param["yl"] - 1):
                Ux2[y, x, t] = Ux[y, x, t] * Ux[y, x, t]
                Uy2[y, x, t] = Uy[y, x, t] * Uy[y, x, t]
                U2[y, x, t] = Ux2[y, x, t] + Uy2[y, x, t]
                if x == param["xl"] - 1 or y == param["yl"] - 1:
                    continue
                dUxdx[y, x, t] = (Ux[y, x + 1, t] - Ux[y, x, t]) / Dx  # du'_x/dx
                dUydy[y, x, t] = (Uy[y + 1, x, t] - Uy[y, x, t]) / Dy  # du'_y/dy

                dUxdx2[y, x, t] = dUxdx[y, x, t] * dUxdx[y, x, t]
                dUydy2[y, x, t] = dUydy[y, x, t] * dUydy[y, x, t]
                dUidxi2[y, x, t] = dUxdx2[y, x, t] + dUydy2[y, x, t]

    # Time average of U^2 and dUx/dx
    Ux2ave = np.nanmean(Ux2, axis=2)
    Uy2ave = np.nanmean(Uy2, axis=2)
    U2ave = np.nanmean(U2, axis=2)
    dUxdx2ave = np.nanmean(dUxdx2, axis=2)
    dUydy2ave = np.nanmean(dUydy2, axis=2)
    dUidxi2ave = np.nanmean(dUidxi2, axis=2)

    # Spatial average of time-averaged U^2 and dUx/dx
    # U2ave_spatial = np.nanmean(U2ave)
    # Ux2ave_spatial = np.nanmean(Ux2ave)
    # Uy2ave_spatial = np.nanmean(Uy2ave)
    # dUxidxi2ave_spatial =np.nanmean(dUidxi2ave)
    # dUxdx2ave_spatial = np.nanmean(dUxdx2ave)
    # dUydy2ave_spatial = np.nanmean(dUydy2ave)
    lambdaT = np.sqrt(np.nanmean(U2ave) / np.nanmean(dUidxi2ave))
    lambdaTx = np.sqrt(np.nanmean(Ux2ave) / np.nanmean(dUxdx2ave))
    lambdaTy = np.sqrt(np.nanmean(Uy2ave) / np.nanmean(dUydy2ave))
    Re_lambdaT = np.sqrt(1.0 / 2.0 * np.nanmean(U2ave)) * lambdaT / nu
    Re_lambdaTx = np.sqrt(np.nanmean(Ux2ave))*lambdaTx / nu
    Re_lambdaTy = np.sqrt(np.nanmean(Uy2ave))*lambdaTy / nu

    if save:
        save_computed_results(U2ave, Ux2ave, Uy2ave, dUidxi2ave, dUxdx2ave, dUydy2ave, savedir, save)

    return lambdaT, Re_lambdaT, lambdaTx, Re_lambdaTx, lambdaTy, Re_lambdaTy

def plot_lambda_and_Relambda(filepath, lambdaT, Re_lambdaT, lambdaTx, Re_lambdaTx, lambdaTy, Re_lambdaTy, savedir, save=False):
    """
    Plots Taylor microscale (lambda) and Re_lambda using Ux and Uy, and save the plots in savedir

    INPUT
    -----
    filepath : str
        definite path to the h5 file from JHTD
    lambdaT, Re_lambdaT, lambdaTx, Re_lambdaTx, lambdaTy, Re_lambdaTy: 3d numpy arrays, obtained from compute_lambda()
    savedir: str
    save: bool
        True if you would like to save figures in savedir

    OUTPUT
    ------
    None
    """
    # Read parameters from the filename
    param = jhtd_get.get_parameters(filepath)
    # Make 1d arrays for time and coordinates
    T,X,Y,Z = generate_XYZT_arrays(param)

    # Calculate averages of lamba and Re_lambda
    Re_lambdaTave = np.nanmean(Re_lambdaT)
    lambdaTave = np.nanmean(lambdaT)
    Re_lambdaTxave = np.nanmean(Re_lambdaTx)
    lambdaTxave = np.nanmean(lambdaTx)
    Re_lambdaTyave = np.nanmean(Re_lambdaTy)
    lambdaTyave = np.nanmean(lambdaTy)


    # Reduce 3d numpy arrays to 2d numpy arrays
    lambdaTForPlot=np.squeeze(lambdaT, axis=2)
    Re_lambdaTForPlot=np.squeeze(Re_lambdaT, axis=2)
    lambdaTxForPlot=np.squeeze(lambdaTx, axis=2)
    Re_lambdaTxForPlot=np.squeeze(Re_lambdaTx, axis=2)
    lambdaTyForPlot=np.squeeze(lambdaTy, axis=2)
    Re_lambdaTyForPlot=np.squeeze(Re_lambdaTy, axis=2)


    print ('Plotting Local Taylor microscale color maps...')

    lambdaMax=0.5
    Re_lambdaMax=1000.0


    ## Plot Local Taylor miroscale
    #Local lambda
    plt.figure(figsize=(32,24))
    plt.pcolor(X, Y, lambdaTForPlot, cmap='RdBu',label='z=0',vmin=0,vmax=lambdaMax)
    cbar1=plt.colorbar()
    cbar1.ax.set_ylabel('Local $\lambda$', fontsize=75)
    cbar1.ax.tick_params(labelsize=50)
    #plt.title('$\lambda$',fontsize=75)
    plt.xlabel('X',fontsize=75)
    plt.ylabel('Y',fontsize=75)
    plt.xticks(size=50)
    plt.yticks(size=50)
    if save:
        filename = 'lambdaLocal_z=0_t=0_ave{0:.3f}'.format(lambdaTave)
        filepath = savedir + filename
        graph.save(filepath, ext='png')

    #Local lambda_x
    plt.figure(figsize=(32,24))
    plt.pcolor(X, Y, lambdaTxForPlot, cmap='RdBu',label='z=0',vmin=0,vmax=lambdaMax)
    cbar1=plt.colorbar()
    cbar1.ax.set_ylabel('Local $\lambda_x$', fontsize=75)
    cbar1.ax.tick_params(labelsize=50)
    #plt.title('$\lambda$',fontsize=75)
    plt.xlabel('X',fontsize=75)
    plt.ylabel('Y',fontsize=75)
    plt.xticks(size=50)
    plt.yticks(size=50)
    if save:
        filename = 'lambdaxLocal_z=0_t=0_ave{0:.3f}'.format(lambdaTxave)
        filepath = savedir + filename
        graph.save(filepath, ext='png')

    #Local lambda_y
    plt.figure(figsize=(32,24))
    plt.pcolor(X, Y, lambdaTyForPlot, cmap='RdBu',label='z=0',vmin=0,vmax=lambdaMax)
    cbar1=plt.colorbar()
    cbar1.ax.set_ylabel('Local $\lambda_y$', fontsize=75)
    cbar1.ax.tick_params(labelsize=50)
    plt.xlabel('X',fontsize=75)
    plt.ylabel('Y',fontsize=75)
    plt.xticks(size=50)
    plt.yticks(size=50)
    if save:
        filename = 'lambdayLocal_z=0_t=0_ave{0:.3f}'.format(lambdaTyave)
        filepath = savedir + filename
        graph.save(filepath, ext='png')



    print ('Plotting Local Re_lambda color maps...')
    ## Plot Local Re_lambda
    #Local Re_lambda
    plt.figure(figsize=(32,24))
    plt.pcolor(X, Y, Re_lambdaTForPlot, cmap='RdBu',label='z=0', vmin=0,vmax=Re_lambdaMax)
    cbar2=plt.colorbar()
    cbar2.ax.set_ylabel('Local $Re_\lambda$', fontsize=75)
    cbar2.ax.tick_params(labelsize=50)
    plt.xlabel('X',fontsize=75)
    plt.ylabel('Y',fontsize=75)
    plt.xticks(size=50)
    plt.yticks(size=50)
    if save:
        filename = 'RelambdaLocal_z=0_t=0_ave{0:.0f}'.format(Re_lambdaTave)
        filepath = savedir + filename
        graph.save(filepath, ext='png')

    #Local Re_lambda_x
    plt.figure(figsize=(32,24))
    plt.pcolor(X, Y, Re_lambdaTxForPlot, cmap='RdBu',label='z=0', vmin=0,vmax=Re_lambdaMax)
    cbar2=plt.colorbar()
    cbar2.ax.set_ylabel('$Local Re_{\lambda_x}$', fontsize=75)
    cbar2.ax.tick_params(labelsize=50)
    plt.xlabel('X',fontsize=75)
    plt.ylabel('Y',fontsize=75)
    plt.xticks(size=50)
    plt.yticks(size=50)
    if save:
        filename = 'RelambdaxLocal_z=0_t=0_ave{0:.0f}'.format(Re_lambdaTxave)
        filepath = savedir + filename
        graph.save(filepath, ext='png')

    #Local Re_lambda_y
    plt.figure(figsize=(32,24))
    plt.pcolor(X, Y, Re_lambdaTyForPlot, cmap='RdBu',label='z=0', vmin=0,vmax=Re_lambdaMax)
    cbar2=plt.colorbar()
    cbar2.ax.set_ylabel('Local $Re_{\lambda_y}$', fontsize=75)
    cbar2.ax.tick_params(labelsize=50)
    plt.xlabel('X',fontsize=75)
    plt.ylabel('Y',fontsize=75)
    plt.xticks(size=50)
    plt.yticks(size=50)
    if save:
        filename = 'RelambdayLocal_z=0_t=0_ave{0:.0f}'.format(Re_lambdaTyave)
        filepath = savedir + filename
        graph.save(filepath, ext='png')

    print 'Done'

def save_computed_results(U2ave, Ux2ave, Uy2ave, dUidxi2ave, dUxdx2ave, dUydy2ave, savedir, save=True):
    """
    Generates a txt file in which the computation results will be written (U2_ave, lambda, Re_lambda)

    Parameters
    ----------
    U2ave: 2-dim numpy array, time-averaged values of U2(x,y)=Ux2(x,y)+Uy(x,y)
    Ux2ave: 2-dim numpy array, time-averaged values of Ux2(x,y)
    Uy2ave: 2-dim numpy array, time-averaged values of Uy2(x,y)
    dUidxi2ave: 2-dim numpy array, time-averaged values of (dUx/dx)^(2) + (dUy/dy)^(2)
    dUxdx2ave: 2-dim numpy array, time-averaged values of (dUx/dx)^(2)
    dUydy2ave: 2-dim numpy array, time-averaged values of (dUy/dy)^(2)
    savedir: str, path to a directory where the output file will be stored
    save: bool, If True, the data will be saved in a txt file, and will be printed. Otherwise, it will only print the computed results

    Returns
    -------

    """
    U2ave_spatial = np.nanmean(U2ave)
    Ux2ave_spatial = np.nanmean(Ux2ave)
    Uy2ave_spatial = np.nanmean(Uy2ave)
    dUxidxi2ave_spatial =np.nanmean(dUidxi2ave)
    dUxdx2ave_spatial = np.nanmean(dUxdx2ave)
    dUydy2ave_spatial = np.nanmean(dUydy2ave)
    lambdaT = np.sqrt(np.nanmean(U2ave) / np.nanmean(dUidxi2ave))
    lambdaTx = np.sqrt(np.nanmean(Ux2ave) / np.nanmean(dUxdx2ave))
    lambdaTy = np.sqrt(np.nanmean(Uy2ave) / np.nanmean(dUydy2ave))
    Re_lambda = np.sqrt(1.0 / 2.0 * np.nanmean(U2ave)) * np.sqrt(np.nanmean(U2ave) / np.nanmean(dUidxi2ave)) / nu
    Re_lambdax = np.sqrt(np.nanmean(Ux2ave))*np.sqrt(np.nanmean(Ux2ave)/np.nanmean(dUxdx2ave))/nu
    Re_lambday = np.sqrt(np.nanmean(Uy2ave))*np.sqrt(np.nanmean(Uy2ave)/np.nanmean(dUydy2ave))/nu
    if save:
        filename = "Computation_results.txt"
        filepath = savedir + filename
        f = open(filepath,'w')
        f.write('Spatial average: at t=t0:\n')
        f.write('U2ave, dUidxi2ave, lambda=sqrt(U2ave/dUidxi2ave), and Re_lambda \n')
        f.write(str(U2ave_spatial) + ', ' + str(dUxidxi2ave_spatial) + ', '  +  str(lambdaT) + ', ' + str(Re_lambda) + '\n' )
        f.write('Ux2ave and dUxdx2ave, lambdax=sqrt(Ux2ave/dUxdx2ave), and Re_lambdax \n')
        f.write(str(Ux2ave_spatial) + ', ' + str(dUxdx2ave_spatial) + ', '  +  str(lambdaTx) + ', ' + str(Re_lambdax) + '\n')
        f.write('Uy2ave and dUxdx2ave, lambdax=sqrt(Uy2ave/dUydy2ave), and Re_lambday \n')
        f.write(str(Uy2ave_spatial) + ', ' + str(dUydy2ave_spatial) + ', '  +  str(lambdaTy) + ', ' + str(Re_lambday) + '\n')
        f.close()

    print '------------------------------------------------------'
    print 'lambda and Re_lambda:'
    print 'U2ave, dUidxi2ave, lambda=sqrt(U2ave/dUidxi2ave), and Re_lambda'
    print U2ave_spatial, dUxidxi2ave_spatial, lambdaT, Re_lambda
    print 'Ux2ave and dUxdx2ave, lambdax=sqrt(Ux2ave/dUxdx2ave), and Re_lambdax'
    print Ux2ave_spatial, dUxdx2ave_spatial, lambdaTx, Re_lambdax
    print 'Uy2ave and dUydy2ave, lambday=sqrt(Uy2ave/dUidxi2ave), and Re_lambday'
    print Uy2ave_spatial, dUydy2ave_spatial, lambdaTy, Re_lambday
    print '------------------------------------------------------'

def save_computed_results_Local(lambdaT_Local, Re_lambdaT_Local, lambdaTx_Local, Re_lambdaTx_Local, lambdaTy_Local, Re_lambdaTy_Local, savedir, save=True):
    """
    Parameters
    ----------
    lambdaT_Local: 2d numpy array, lambdaT_Local= u'(x,y)^2 / (du'i/dxi)^2 where u'^2 = ux^2 + uy^2, (du'i/dxi)^2 = (dux/dx)^2 + (duy/dy)^2
    Re_lambdaT_Local: 2d numpy array, Re_lambdaT_Local(x,y) = (u'(x,y)/2)*lambdaT_local(x,y)/nu
    lambdaTx_Local: 2d numpy array, lambdaTx_Local= ux'^2 / (dux'/dx)^2
    Re_lambdaTx_Local: 2d numpy array, Re_lambdaTx_Local(x,y) = (ux'(x,y)/2)*lambdaT_local(x,y)/nu
    lambdaTy_Local: 2d numpy array, lambdaTy_Local= uy'^2 / (dux'/dx)^2
    Re_lambdaTy_Local: 2d numpy array, Re_lambdaTy_Local(x,y) = (ux'(x,y)/2)*lambdaT_local(x,y)/nu
    savedir: str, path to a directory where the output file will be stored
    save: bool, If True, the data will be saved in a txt file, and will be printed. Otherwise, it will only print the computed results

    Returns
    -------

    """

    Re_lambdaTave = np.nanmean(Re_lambdaT_Local)
    lambdaTave = np.nanmean(lambdaT_Local)
    Re_lambdaTxave = np.nanmean(Re_lambdaTx_Local)
    lambdaTxave = np.nanmean(lambdaTx_Local)
    Re_lambdaTyave = np.nanmean(Re_lambdaTy_Local)
    lambdaTyave = np.nanmean(lambdaTy_Local)
    if save:
        filename = "Computation_results_local.txt"
        filepath = savedir + filename
        f = open(filepath, 'w')
        f.write('Spatial average: at t=t0:\n')
        f.write('lambda (averaged Local lambda), Re_lambda (averaged Local Re_lambda)\n')
        f.write(str(lambdaTave) + ', ' + str(Re_lambdaTave) + '\n')
        f.write('lambda_x (averaged Local lambda_x), Re_lambda_x (averaged Local Re_lambda_x)\n')
        f.write(str(lambdaTxave) + ', ' + str(Re_lambdaTxave) + '\n')
        f.write('lambda_y (averaged Local lambda_y), Re_lambda_y (averaged Local Re_lambda_y)\n')
        f.write(str(lambdaTyave) + ', ' + str(Re_lambdaTyave) + '\n')
        f.close()

    print '------------------------------------------------------'
    print 'Average of Local lambda and Re_lambda'
    print 'Re_lambda (averaged Local Re_lambda): ' + str(Re_lambdaTave)
    print 'lambda (averaged Local lambda): ' + str(lambdaTave)
    print 'Re_lambda_x (averaged Local Re_lambda_x): ' + str(Re_lambdaTxave)
    print 'lambda_x (averaged Local lambda_x): ' + str(lambdaTxave)
    print 'Re_lambda_y (averaged Local Re_lambda_y): ' + str(Re_lambdaTyave)
    print 'lambda_y (averaged Local lambda_y): ' + str(lambdaTyave)
    print '------------------------------------------------------'

