import sys
import operator
import numpy as np
import xarray as xr
import numpy as np
import numpy.ma as ma
import dask.array as dsar
from dask import delayed
from functools import reduce
import scipy.signal as sps
import scipy.linalg as spl
import scipy.signal as signal
import xscale.spectral.fft as xfft
import PowerSpec as ps
    
    
########################################            
def get_dx_dy(data,navlon,navlat):
    # Obtain dx and dy
    x,y,dat = ps.interpolate(data,np.array(navlon),np.array(navlat))
    x1,y1 = x[0,:],y[:,0]
    dx=np.int(np.ceil(x1[1]-x1[0]))
    dy=np.int(np.ceil(y1[1]-y1[0]))
    dx = dx/1E3
    dy = dy/1E3
    return dx,dy        

#########################################
def apply_window(da, dims, window_type='hanning'):
    """Creating windows in dimensions dims."""

    if window_type not in ['hanning']:
        raise NotImplementedError("Only hanning window is supported for now.")

    numpy_win_func = getattr(np, window_type)

    if da.chunks:
        def dask_win_func(n):
            return dsar.from_delayed(
                delayed(numpy_win_func, pure=True)(n),
                (n,), float)
        win_func = dask_win_func
    else:
        win_func = numpy_win_func

    windows = [xr.DataArray(win_func(len(da[d])),
               dims=da[d].dims, coords=da[d].coords) for d in dims]

    return da * reduce(operator.mul, windows[::-1])

#########################################
def get_wavnum_kradial(kx,ky):
    ''' Compute a wavenumber vector  '''
    k, l = np.meshgrid(kx,ky)
    kradial = np.sqrt(k**2 + l**2)
    kmax = np.sqrt((k.max())**2 + (l.max())**2)/np.sqrt(2)
    
    dkx = np.abs(kx[2]-kx[1])
    dky = np.abs(ky[2]-ky[1])
    dkradial = min(dkx,dky)
    
    # radial wavenumber
    wavnum = (dkradial.data)*np.arange(1,int(kmax/dkradial))
    return wavnum,kradial

#########################################
def get_spec_1D(kradial,wavnum,spec_2D):
    ''' Compute the azimuthaly avearge of the 2D spectrum '''
    spec_1D = np.zeros(len(wavnum))
    for i in range(wavnum.size):
        kfilt =  (kradial>=wavnum[i] - wavnum[0]) & (kradial<=wavnum[i])
        N = kfilt.sum()
        spec_1D[i] = (spec_2D[kfilt].sum())*wavnum[i]/N  #
    return spec_1D

#########################################
def get_f_kx_ky(hat):
    f = hat.f_time_counter
    kx = hat.f_x
    ky = hat.f_y
    return f,kx,ky

#########################################
def get_f_kx_ky_mit(hat):
    f = hat.f_time
    kx = hat.f_i
    ky = hat.f_j
    return f,kx,ky

#########################################
def get_f_kx_ky_flo(hat):
    f = hat.f_time
    kx = hat.f_x
    ky = hat.f_y
    return f,kx,ky

#########################################
def get_f_kx_ky_jet(hat):
    f = hat.f_time_counter
    kx = hat.f_x_rho
    ky = hat.f_y_rho
    return f,kx,ky


#########################################
def get_f_k_in_2D(kradial,wavnum,spec2D):
    _spec_1D = []
    for i in range(len(spec2D)):
        if i%100==0: print(i)
        psd_2D = spec2D[i]
        spec1D = get_spec_1D(kradial,wavnum,psd_2D)
        _spec_1D.append(spec1D)
    spec_1D = np.array(_spec_1D)
    return spec_1D

#########################
def get_flux(wavnum2D,wavnum1D,spec_2D):
    ''' Compute KE flux'''
    flux = np.zeros(len(wavnum1D))
    for i in range(wavnum1D.size):
        kfilt =  (wavnum1D[i] <= wavnum2D ) 
        flux[i] = (spec_2D[kfilt]).sum()
    return flux

#########################
def get_flux_in_1D(kradial,wavnum,spec2D):
    _flux_1D = []
    for i in range(len(spec2D)):
        if i%100==0: print(i)
        psd_2D = spec2D[i]
        flux1D = get_flux(kradial,wavnum,psd_2D)
        _flux_1D.append(flux1D)
    flux_1D = np.array(_flux_1D)
    return flux_1D



#########################################
def detrendn(da, axes=None):
    """
    Detrend by subtracting out the least-square plane or least-square cubic fit
    depending on the number of axis.
    Parameters
    ----------
    da : `dask.array`
        The data to be detrended
    Returns
    -------
    da : `numpy.array`
        The detrended input data
    """
#     if da.ndim > 2:
#         raise ValueError('The data should only have two dimensions')
#     print(da.shape)
    N = [da.shape[n] for n in axes]
    M = []
    for n in range(da.ndim):
        if n not in axes:
            M.append(da.shape[n])

    if len(N) == 2:
        G = np.ones((N[0]*N[1],3))
        for i in range(N[0]):
            G[N[1]*i:N[1]*i+N[1], 1] = i+1
            G[N[1]*i:N[1]*i+N[1], 2] = np.arange(1, N[1]+1)
        if type(da) == xr.DataArray:
            d_obs = np.reshape(da.copy().values, (N[0]*N[1],1))
        else:
            d_obs = np.reshape(da.copy(), (N[0]*N[1],1))
    elif len(N) == 3:
        if type(da) == xr.DataArray:
            if da.ndim > 3:
                raise NotImplementedError("Cubic detrend is not implemented "
                                         "for 4-dimensional `xarray.DataArray`."
                                         " We suggest converting it to "
                                         "`dask.array`.")
            else:
                d_obs = np.reshape(da.copy().values, (N[0]*N[1]*N[2],1))
        else:
            d_obs = np.reshape(da.copy(), (N[0]*N[1]*N[2],1))

        G = np.ones((N[0]*N[1]*N[2],4))
        G[:,3] = np.tile(np.arange(1,N[2]+1), N[0]*N[1])
        ys = np.zeros(N[1]*N[2])
        for i in range(N[1]):
            ys[N[2]*i:N[2]*i+N[2]] = i+1
        G[:,2] = np.tile(ys, N[0])
        for i in range(N[0]):
            G[len(ys)*i:len(ys)*i+len(ys),1] = i+1
    else:
        raise NotImplementedError("Detrending over more than 4 axes is "
                                 "not implemented.")

    m_est = np.dot(np.dot(spl.inv(np.dot(G.T, G)), G.T), d_obs)
    d_est = np.dot(G, m_est)

    lin_trend = np.reshape(d_est, da.shape)

    return da - lin_trend


def velocity_derivatives(u, v, xdim, ydim, dx):

    uhat = xfft.fft(u, dim=(xdim, ydim), dx=dx, sym=True)
    vhat = xfft.fft(v, dim=(xdim, ydim), dx=dx, sym=True)
    k = uhat['f_%s' % xdim]
    l = vhat['f_%s' % ydim]
    u_x_hat = uhat * 2 * np.pi * 1j * k 
    u_y_hat = uhat * 2 * np.pi * 1j * l 
    v_x_hat = vhat * 2 * np.pi * 1j * k
    v_y_hat = vhat * 2 * np.pi * 1j * l
    ds_derivatives = xr.Dataset({'u_x': xfft.ifft(u_x_hat),
                                 'u_y': xfft.ifft(u_y_hat),
                                 'v_x': xfft.ifft(v_x_hat),
                                 'v_y': xfft.ifft(v_y_hat)
                                })
    return ds_derivatives
