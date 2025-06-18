#arXiv:2008.06481
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

def precalculatedKcoeffs(Ndim, path2kernels):
    fftdim = 2*Ndim-1;
    kernel = np.fromfile(path2kernels+'KernelD'+str(Ndim)+'.dat', dtype=np.complex128)
    kernel = (np.flip(kernel)).reshape((fftdim,Ndim,Ndim))
    return kernel
    
def PSrepresentationFourierCoeff(rho, Kcoeffs):
    Ndim = len(rho)
    fftdim = 2*Ndim-1;
    result = np.zeros((fftdim,fftdim),dtype=np.complex128)
    #This is an implementation using the offset trace operation
    #of the element-wise product of matrices
    for l in range(0, fftdim):
        prod = np.multiply(rho, Kcoeffs[l])
        for m in range(0, fftdim):
            result[l,m] = np.trace(prod,m-Ndim+1)
    return result


def PSrepresentationFourierCoeff_LOOP(rho, Kcoeffs):
    Ndim = len(rho)
    fftdim = 2*Ndim-1;
    result = np.zeros((fftdim,fftdim),dtype=np.complex128)
    #This is straightforwad loop-based implementation which might be slow
    for l in range(0, fftdim):
        for m in range(0, fftdim):
            pres = 0
            lowb = max(Ndim - m - 1, 0)
            upb  = min(2*Ndim - m -1, Ndim)
            for lam in range(lowb, upb):
                pres = pres + rho[lam,lam+m-Ndim+1]*Kcoeffs[l,lam,lam+m-Ndim+1]
            result[l,m] = pres
    return result
    
def PSrepresentationFromFourier(rho, Kcoeffs, finalpoints):
    Ndim = len(rho)
    fourierCoeffs = PSrepresentationFourierCoeff(rho, Kcoeffs)
    fourierCoeffsRef = PSrepresentationFourierCoeff(np.identity(Ndim)/np.sqrt(Ndim), Kcoeffs)

    PSrep = np.fft.ifft2(fourierCoeffs,(2*finalpoints,finalpoints))
    PSrepRef = np.fft.ifft2(fourierCoeffsRef,(2*finalpoints,finalpoints))
    Pi=np.pi
    identityPS= 1/np.sqrt(Ndim-1)
    result = np.real((PSrep/PSrepRef*identityPS).round(12))
    result = result[0:finalpoints]
    return result
    
def PSrepPlot(psFunctionArray, filename):
    ny, nx = psFunctionArray.shape

    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(0, np.pi, ny)

    xv, yv = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(60, 30)
    ax.plot_surface(xv, yv, psFunctionArray, cmap='inferno', rstride=1, cstride=1, alpha=None, antialiased=True)
    plt.savefig(filename)


def PSrepPlot_sphere(psFunctionArray):#, filename):
    #psFunctionArray = value at x=phi,y=theta
    ny, nx = psFunctionArray.shape

    phi_plot = np.linspace(0, 2*np.pi, nx) #x
    theta_plot = np.linspace(0, np.pi, ny) #y

    phi_plot,theta_plot=np.meshgrid(phi_plot,theta_plot)
    
    x=10*np.sin(theta_plot)*np.cos(phi_plot)
    y=10*np.sin(theta_plot)*np.sin(phi_plot)
    z=10*np.cos(theta_plot)

    norm=colors.Normalize(vmin = -np.max(np.abs(psFunctionArray)),
                      vmax = np.max(np.abs(psFunctionArray)), clip = False)
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(20,20)#60, 30)

    ax.plot_surface(x, y, z, cmap=cm.coolwarm, rstride=1, cstride=1,
                       linewidth=0, antialiased=False,
                       facecolors=cm.coolwarm(norm(psFunctionArray)))

    #ax.plot_surface(xv, yv, psFunctionArray, cmap='inferno', rstride=1, cstride=1, alpha=None, antialiased=True)
    plt.show()#savefig(filename)

def PSrepPlot_plane(psFunctionArray,title):#, filename):
    #psFunctionArray = value at x=phi,y=theta
    ny, nx = psFunctionArray.shape

    phi_plot = np.linspace(0, 2*np.pi, nx) #x
    theta_plot = np.linspace(0, np.pi, ny) #y

    phi_plot,theta_plot=np.meshgrid(phi_plot,theta_plot)
    
    x=10*np.sin(theta_plot)*np.cos(phi_plot)
    y=10*np.sin(theta_plot)*np.sin(phi_plot)
    z=10*np.cos(theta_plot)

    vmin = -np.max(np.abs(psFunctionArray))
    vmax = np.max(np.abs(psFunctionArray))
    

    fig = plt.figure()
    ax = fig.add_subplot(111)#, projection='3d')
    #ax.view_init(20,20)#60, 30)

    im = ax.pcolor( phi_plot/np.pi,theta_plot/np.pi,psFunctionArray, cmap=cm.coolwarm,vmin=vmin,vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(r'$\phi/\pi$')
    ax.set_ylabel(r'$\theta/\pi$')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Phase space distribution')

    ax.set_title(title)
    ax.set_xlabel(r'$\phi/\pi$')
    ax.set_ylabel(r'$\theta/\pi$')

    #ax.plot_surface(xv, yv, psFunctionArray, cmap='inferno', rstride=1, cstride=1, alpha=None, antialiased=True)
    plt.show()#savefig(filename)

def PSrepPlot_plane_save(psFunctionArray,title,filename):#, filename):
    #psFunctionArray = value at x=phi,y=theta
    ny, nx = psFunctionArray.shape

    phi_plot = np.linspace(0, 2*np.pi, nx) #x
    theta_plot = np.linspace(0, np.pi, ny) #y

    phi_plot,theta_plot=np.meshgrid(phi_plot,theta_plot)
    
    x=10*np.sin(theta_plot)*np.cos(phi_plot)
    y=10*np.sin(theta_plot)*np.sin(phi_plot)
    z=10*np.cos(theta_plot)

    vmin = -np.max(np.abs(psFunctionArray))
    vmax = np.max(np.abs(psFunctionArray))
    

    fig = plt.figure()
    ax = fig.add_subplot(111)#, projection='3d')
    #ax.view_init(20,20)#60, 30)

    im = ax.pcolor( phi_plot/np.pi,theta_plot/np.pi,psFunctionArray, cmap=cm.coolwarm,vmin=vmin,vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Phase space distribution')

    ax.set_title(title)
    ax.set_xlabel(r'$\phi/\pi$')
    ax.set_ylabel(r'$\theta/\pi$')

    #ax.plot_surface(xv, yv, psFunctionArray, cmap='inferno', rstride=1, cstride=1, alpha=None, antialiased=True)
    plt.savefig(filename)
