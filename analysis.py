#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 18:48:38 2019

@author: leguillou
"""

import argparse 
import sys, os
import xarray as xr
import numpy as np 
from PowerSpec import wavenumber_spectra
import pickle
from scipy import interpolate
import Wavenum_freq_spec_func as wfs
import xscale.spectral.fft as xfft

def read_data(path_data, prods, bc, time_offset,DUACS=True):
    
    data = {}
    ds = xr.open_dataset(path_data+'/data_interpolated.nc')
    # Time and grid
    data['time'] = ds['time'][time_offset:]
    data['lon'] = ds['lon'][bc:-(bc+1),bc:-(bc+1)]
    data['lat'] = ds['lat'][bc:-(bc+1),bc:-(bc+1)]
    # Variables
    data['ref'] = {}
    if DUACS:
        data['duacs'] = {}
    data['da'] = {}
    for prod in prods:
        data['ref'][prod] = ds[prod+'_ref'][time_offset:,bc:-(bc+1),bc:-(bc+1)]
        if DUACS:
            data['duacs'][prod] = ds[prod+'_duacs'][time_offset:,bc:-(bc+1),bc:-(bc+1)]
        data['da'][prod] = ds[prod+'_da'][time_offset:,bc:-(bc+1),bc:-(bc+1)]
    
    return data



def ana_spec(data, prods,DUACS=True,r=0.5):
    
    if DUACS:
        PSD = {'ref':{},'duacs':{},'da':{}}
        Score = {'duacs':{},'da':{}}
        Num = {'duacs':{},'da':{}}
    else:
        PSD = {'ref':{},'da':{}}
        Score = {'da':{}}
        Num = {'da':{}}

    lon = data['lon']
    lat = data['lat']
    
    # Loop on variables 
    for prod in prods:
        # Initialize lists
        wavenumber, PSD_ref = ana_compute_spec(data['ref'][prod].values,lon,lat)
        PSD['ref'][prod] = PSD_ref
        if DUACS:
            PSD['duacs'][prod] = ana_compute_spec(data['duacs'][prod].values,lon,lat)[1]
            PSD_err_duacs = ana_compute_spec(data['duacs'][prod]-data['ref'][prod].values,lon,lat)[1]
            Score['duacs'][prod] = 1 - PSD_err_duacs / PSD_ref
            f = interpolate.interp1d(Score['duacs'][prod], 1/wavenumber, axis=0)  
            if np.max(Score['duacs'][prod])>r and np.min(Score['duacs'][prod])<r:
                Num['duacs'][prod] = f(r)
            else:
                Num['duacs'][prod] = np.nan

        PSD['da'][prod] = ana_compute_spec(data['da'][prod].values,lon,lat)[1]
        PSD_err_da = ana_compute_spec(data['da'][prod].values-data['ref'][prod].values,lon,lat)[1]
        Score['da'][prod] = 1 - PSD_err_da / PSD_ref
        
        f = interpolate.interp1d(Score['da'][prod], 1/wavenumber, axis=0)   
        if np.max(Score['da'][prod])>r and np.min(Score['da'][prod])<r:
            Num['da'][prod] = f(r)
        else:
            Num['da'][prod] = np.nan


    return {'wavenumber':wavenumber, 'PSD':PSD, 'Score':Score, 'Num':Num}




def ana_compute_spec(data,lon,lat,ncentred=20):
    
    psd2D_list = []
    for t in range(data.shape[0]):
        # Compute PSD of the fields and the erros at each timestamp
        wavenumber, psd2D = wavenumber_spectra(np.ma.array(data[t,ncentred:-(ncentred+1),ncentred:-(ncentred+1)]),lon[ncentred:-(ncentred+1),ncentred:-(ncentred+1)],lat[ncentred:-(ncentred+1),ncentred:-(ncentred+1)]) 
        psd2D_list.append(psd2D)
        
    return wavenumber,np.mean(psd2D_list,axis=0)




def ana_rmse(data, prods, DUACS=True):
    
    if DUACS:
        RMSE = {'duacs':{},'da':{}}
        Score = {'duacs':{},'da':{}}
        Num = {'duacs':{},'da':{}}
    else:
        RMSE = {'da':{}}
        Score = {'da':{}}
        Num = {'da':{}}
    
    for prod in prods:  
        NT,NY,NX = data['ref'][prod].shape   
        if DUACS:
            RMSE['duacs'][prod] = ana_compute_rmse(data['ref'][prod].values, data['duacs'][prod].values)
            Score['duacs'][prod] = 1 - RMSE['duacs'][prod]/np.std(data['ref'][prod].values)
            Num['duacs'][prod] = np.mean(Score['duacs'][prod])
        RMSE['da'][prod] = ana_compute_rmse(data['ref'][prod].values, data['da'][prod].values)
        Score['da'][prod] = 1 - RMSE['da'][prod]/np.std(data['ref'][prod].values)        
        Num['da'][prod] = np.mean(Score['da'][prod])
        
    return {'time':data['time'], 'RMSE':RMSE, 'Score':Score, 'Num':Num}




def ana_compute_rmse(reference, DAprod, normalized=False,ncentred=None):  
    """
    NAME 
        diag_computing_rmse

    DESCRIPTION Computing RMSE between a reference and a data assimilation product
    
        Args: 
            reference (3d array) : temporal serie of true fields (typically from the NATL60 run)
            DAprod (3d array) : temporal serie of results from a data assimilation experiment. 
            normalized (bool) : if True, the RMSE is normalized. Default is False.
            ncentred (int): lenght of the peripherical band for which RMSE is not computed
            
        Returns: 
            RMSE (1d array) : temporal serie of the RMSE between the two fields.
            
    """
    ntime, nlon, nlat = reference.shape
    RMSE = [np.nan]*ntime
    
    if reference.shape[0] != DAprod.shape[0] : 
        print('Warning : time series are not of the same lenght. We take the shorter one. ')
        ntime = min(reference.shape[0],DAprod.shape[0]) - 1
        
    if reference.shape[1:] != DAprod.shape[1:] :
        print('Error : grids are not of the same dimensions. Stop here')
        return  

    for itime in range(ntime):
        
        if normalized:
            if ncentred is not None:
                rmse = np.sqrt(np.sum(np.sum(np.square(DAprod[itime,ncentred:-(ncentred+1),ncentred:-(ncentred+1)]-reference[itime,ncentred:-(ncentred+1),ncentred:-(ncentred+1)])))/nlon/nlat) / \
                    ( np.max(np.max(reference[itime,ncentred:-(ncentred+1),ncentred:-(ncentred+1)]))-np.min(np.min(reference[itime,ncentred:-(ncentred+1),ncentred:-(ncentred+1)])) )
            else:
                rmse = np.sqrt(np.sum(np.sum(np.square(DAprod[itime,:,:]-reference[itime,:,:])))/nlon/nlat) / ( np.max(np.max(reference[itime,:,:]))-np.min(np.min(reference[itime,:,:])) )
        else:
            if ncentred is not None:
                rmse = np.sqrt(np.sum(np.sum(np.square(DAprod[itime,ncentred:-(ncentred+1),ncentred:-(ncentred+1)]-reference[itime,ncentred:-(ncentred+1),ncentred:-(ncentred+1)])))/nlon/nlat)
            else:
                rmse = np.sqrt(np.sum(np.sum(np.square(DAprod[itime,:,:]-reference[itime,:,:])))/nlon/nlat)
            
        RMSE[itime] = rmse
        
    return RMSE

def ana_wk(data,prods,DUACS=True):
    if DUACS:
        WK = {'ref':{},'duacs':{},'da':{}}
        Score = {'duacs':{},'da':{}}
    else:
        WK = {'ref':{},'da':{}}
        Score = {'da':{}}

    lon = data['lon']
    lat = data['lat']
    
    # Loop on variables 
    for prod in prods:
        wavenumber, frequency, WK_ref = compute_wk(data['ref'][prod],lon,lat)
        WK['ref'][prod] = WK_ref
        if DUACS:
            WK['duacs'][prod] = compute_wk(data['duacs'][prod],lon,lat)[2]
            WK_err_duacs = compute_wk(data['duacs'][prod]-data['ref'][prod],lon,lat)[2]
            Score['duacs'][prod] = 1 - WK_err_duacs / WK_ref
        WK['da'][prod] = compute_wk(data['da'][prod],lon,lat)[2]
        WK_err_da = compute_wk(data['da'][prod]-data['ref'][prod],lon,lat)[2]
        Score['da'][prod] = 1 - WK_err_da / WK_ref

    return {'wavenumber':wavenumber, 'frequency':frequency, 'WK':WK, 'Score':Score}


def compute_wk(data,lon,lat):
    
    # - get dx and dy
    dx,dy = wfs.get_dx_dy(data[0],lon,lat)

    #... Detrend data in all dimension ...
    ssh_detrended = wfs.detrendn(data,axes=[0,1,2])

    #... Apply hanning windowing ...') 
    ssh_hanning = wfs.apply_window(ssh_detrended, data.dims, window_type='hanning')

    #... Apply hanning windowing ...') 
    ssh_hat = xfft.fft(ssh_hanning, dim=('time', 'x', 'y'), dx={'x': dx, 'y': dx}, sym=True)

    #... Apply hanning windowing ...') 
    ssh_psd = xfft.psd(ssh_hat)

    #... Get frequency and wavenumber ... 
    frequency = ssh_hat.f_time
    kx = ssh_hat.f_x
    ky = ssh_hat.f_y

    #... Get istropic wavenumber ... 
    wavenumber,kradial = wfs.get_wavnum_kradial(kx,ky)

    #... Get numpy array ... 
    ssh_psd_np = ssh_psd.values

    #... Get 2D frequency-wavenumber field ... 
    SSH_wavenum_freq_spectrum = wfs.get_f_k_in_2D(kradial,wavenumber,ssh_psd_np)
    
    return wavenumber, frequency, SSH_wavenum_freq_spectrum


##======================================================================================================================##
##                MAIN                                                                                                  ##
##======================================================================================================================##

if __name__ == '__main__':
    
    
    #+++++++++++++++++++++++++++++++#
    #    Parsing                    #
    #+++++++++++++++++++++++++++++++#
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path_config_exp', default=None, type=str)        # parameters relative to the DA experiment
    parser.add_argument('--name_config_comp', default=None, type=str)       # parameters relative to NATL60 and DUACS 
    parser.add_argument('--prods', default=['ssh'],nargs='+', type=str)
    parser.add_argument('--overwrite', default=1, type=int)
    # optional parameters that have to be provided if *path_config_exp* is not provided
    parser.add_argument('--name_exp', default=None, type=str)               
    # optional parameters that have to be provided if *name_config_comp* is not provided
    parser.add_argument('--path_out', default=None, type=str)   
    parser.add_argument('--ncentred', default=0, type=int)  
    parser.add_argument('--time_offset', default=0, type=int)  
    parser.add_argument('--DUACS', default=False, type=bool)  
    
    # Parsing
    opts = parser.parse_args()
        
    #+++++++++++++++++++++++++++++++#
    #    GET params                 #
    #+++++++++++++++++++++++++++++++#
    print('\n* Get parameters')
    # parameters relative to the DA experiment
    if opts.path_config_exp is None:
        if opts.name_exp is not None:
            name_exp = opts.name_exp
        else:
            print('Error: either path_config_exp or name_exp has to be specified')
            sys.exit()
    else:        
        dir_exp = os.path.dirname(opts.path_config_exp)
        file_exp = os.path.basename(opts.path_config_exp)
        sys.path.insert(0,dir_exp)
        exp = __import__(file_exp, globals=globals())
        name_exp = exp.name_experiment
        
    # parameters relative to comparison
    if opts.name_config_comp is None:
        if opts.path_out is not None:
            path_out = opts.path_out
            ncentred = opts.ncentred
            time_offset = opts.time_offset
            DUACS = opts.DUACS
        else:
            print('Error: either name_config_comp or path_out has to be specified')
            sys.exit()            
    else:           
        sys.path.insert(0,os.path.join(os.path.dirname(__file__), "configs"))
        comp = __import__(opts.name_config_comp, globals=globals())
        path_out = comp.path_out
        if hasattr(comp, 'path_duacs'):
            DUACS = True
        else:
            print('No DUACS-related parameters --> no comparison with DUACS will be performed')
            DUACS = False
        if hasattr(comp, 'ncentred'):
            ncentred = comp.ncentred
        else:
            print('Warning: argument "ncentred" is not defined in comparison config file. Its value is set to 0')
            ncentred = 0
        if hasattr(comp, 'time_offset'):
            time_offset = comp.time_offset
        else:
            print('Warning: argument "time_offset" is not defined in experiment config file. Its value is set to 0')
            time_offset = 0
    
    #+++++++++++++++++++++++++++++++#
    #    Analysis                   #
    #+++++++++++++++++++++++++++++++#
    file_outputs = path_out+name_exp +'/analysis.pic'
    if os.path.isfile(file_outputs) and opts.overwrite!=1: 
        print(file_outputs, 'already exists')
    else:
        if opts.overwrite==1: 
            print('Warning:', file_outputs, 'already exists but you ask to overwrite it')
        #+++++++++++++++++++++++++++++++#
        #    Read interpolated fields   #
        #+++++++++++++++++++++++++++++++#
        print('\n* Read interpolated fields')
        data = read_data(path_out+name_exp, opts.prods, ncentred, time_offset, DUACS=DUACS)
        
        #+++++++++++++++++++++++++++++++#
        #    RMSE Analysis              #
        #+++++++++++++++++++++++++++++++#
        print('\n* RMSE Analysis')
        rmse = ana_rmse(data, opts.prods, DUACS=DUACS)
        
        #+++++++++++++++++++++++++++++++#
        #    Spectral Analysis          #
        #+++++++++++++++++++++++++++++++#
        print('\n* Spectral Analysis')
        spec = ana_spec(data, opts.prods, DUACS=DUACS)
        
        #+++++++++++++++++++++++++++++++#
        #    WK Analysis          #
        #+++++++++++++++++++++++++++++++#
        print('\n* WK Analysis')
        wk = ana_wk(data, opts.prods, DUACS=DUACS)
        
        #+++++++++++++++++++++++++++++++#
        #    Dump Analysis              #
        #+++++++++++++++++++++++++++++++#
        print('\n* Dump Analysis')   
        print('-->',file_outputs)
        with open(file_outputs, 'wb') as f:
                pickle.dump((rmse, spec, wk), f)
        



    
    
    
    
