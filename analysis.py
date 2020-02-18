#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 18:48:38 2019

@author: leguillou
"""

import argparse 
import sys, os
import netCDF4 as nc
import numpy as np 
from PowerSpec import wavenumber_spectra
import pickle

def read_data(path_data, prods, bc, time_offset,DUACS=True):
    
    data = {}
    ncin = nc.Dataset(path_data+'/data_interpolated.nc')
    # Time and grid
    data['time'] = ncin.variables['time'][time_offset:]
    data['lon'] = ncin.variables['lon'][bc:-(bc+1),bc:-(bc+1)]
    data['lat'] = ncin.variables['lat'][bc:-(bc+1),bc:-(bc+1)]
    # Variables
    data['ref'] = {}
    if DUACS:
        data['duacs'] = {}
    data['da'] = {}
    for prod in prods:
        data['ref'][prod] = ncin.variables[prod+'_ref'][time_offset:,bc:-(bc+1),bc:-(bc+1)]
        if DUACS:
            data['duacs'][prod] = ncin.variables[prod+'_duacs'][time_offset:,bc:-(bc+1),bc:-(bc+1)]
        data['da'][prod] = ncin.variables[prod+'_da'][time_offset:,bc:-(bc+1),bc:-(bc+1)]
    
    return data



def ana_spec(data, prods,DUACS=True):
    
    if DUACS:
        PSD = {'ref':{},'duacs':{},'da':{}}
        recScore = {'duacs':{},'da':{}}
    else:
        PSD = {'ref':{},'da':{}}
        recScore = {'da':{}}
    
    NT = data['time'].size
    lon = data['lon']
    lat = data['lat']
    
    # Loop on variables 
    for prod in prods:
        # Initialize lists
        PSD['ref'][prod] = []
        if DUACS:
            PSD['duacs'][prod] = []
        PSD['da'][prod] = []
        PSD_err_duacs = []
        PSD_err_da = []
        # Time loop
        for t in range(NT):
            # Compute PSD of the fields and the erros at each timestamp
            wavenumber, psd2D_ref_t = wavenumber_spectra(np.ma.array(data['ref'][prod][t]),lon,lat) 
            if DUACS:
                wavenumber, psd2D_duacs_t = wavenumber_spectra(np.ma.array(data['duacs'][prod][t]),lon,lat) 
                wavenumber, psd2D_err_duacs_t = wavenumber_spectra(np.ma.array(data['duacs'][prod][t]-data['ref'][prod][t]),lon,lat) 
            wavenumber, psd2D_da_t = wavenumber_spectra(np.ma.array(data['da'][prod][t]),lon,lat)             
            wavenumber, psd2D_err_da_t = wavenumber_spectra(np.ma.array(data['da'][prod][t]-data['ref'][prod][t]),lon,lat)            
            # Append to list
            PSD['ref'][prod].append(psd2D_ref_t)
            if DUACS:
                PSD['duacs'][prod].append(psd2D_duacs_t)
            PSD['da'][prod].append(psd2D_da_t)
            PSD_err_duacs.append(psd2D_err_duacs_t)
            PSD_err_da.append(psd2D_err_da_t)                
        # Average temporally 
        PSD['ref'][prod] = np.mean(PSD['ref'][prod],axis=0)
        if DUACS:
            PSD['duacs'][prod] = np.mean(PSD['duacs'][prod],axis=0)
            recScore['duacs'][prod] = 1 - np.mean(PSD_err_duacs,axis=0) / PSD['ref'][prod]
        PSD['da'][prod] = np.mean(PSD['da'][prod],axis=0)        
        recScore['da'][prod] = 1 - np.mean(PSD_err_da,axis=0) / PSD['ref'][prod]

    return {'wavenumber':wavenumber,'PSD':PSD, 'recScore':recScore}




def ana_rmse(data, prods, DUACS=True):
    if DUACS:
        RMSE = {'duacs':{},'da':{}}
    else:
        RMSE = {'da':{}}
    
    for prod in prods:  
        NT,NY,NX = data['ref'][prod].shape   
        if DUACS:
            RMSE['duacs'][prod] = [np.sqrt(np.sum(np.sum(np.square(data['duacs'][prod][t,:,:]-data['ref'][prod][t,:,:])))/NY/NX) for t in range(NT)]
        RMSE['da'][prod] = [np.sqrt(np.sum(np.sum(np.square(data['da'][prod][t,:,:]-data['ref'][prod][t,:,:])))/NY/NX) for t in range(NT)]
        
    return {'time':data['time'], 'RMSE':RMSE}




##======================================================================================================================##
##                MAIN                                                                                                  ##
##======================================================================================================================##

if __name__ == '__main__':
    #+++++++++++++++++++++++++++++++#
    #    _#
    #+++++++++++++++++++++++++++++++#
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path_config_exp', default=None, type=str)     # parameters relative to the DA experiment
    parser.add_argument('--name_config_comp', default=None, type=str)    # parameters relative to NATL60 and DUACS 
    parser.add_argument('--prods', default=['ssh'],nargs='+', type=str)
    parser.add_argument('--overwrite', default=1, type=int)
    opts = parser.parse_args()
        
    #+++++++++++++++++++++++++++++++#
    #    GET params                 #
    #+++++++++++++++++++++++++++++++#
    print('\n* Get parameters')
    # parameters relative to the DA experiment
    dir_exp = os.path.dirname(opts.path_config_exp)
    name_exp = os.path.basename(opts.path_config_exp)
    sys.path.insert(0,dir_exp)
    exp = __import__(name_exp, globals=globals())
    # parameters relative to NATL60 and DUACS 
    sys.path.insert(0,os.path.join(os.path.dirname(__file__), "configs"))
    comp = __import__(opts.name_config_comp, globals=globals())
    
    #+++++++++++++++++++++++++++++++#
    #    Analysis                   #
    #+++++++++++++++++++++++++++++++#
    file_outputs = comp.path_out+exp.name_experiment +'/analysis.pic'
    if os.path.isfile(file_outputs) and opts.overwrite!=1: 
        print(file_outputs, 'already exists')
    else:
        if opts.overwrite==1: 
            print('Warning:', file_outputs, 'already exists but you ask to overwrite it')
        #+++++++++++++++++++++++++++++++#
        #    Read interpolated fields   #
        #+++++++++++++++++++++++++++++++#
        # DUACS
        if hasattr(comp, 'path_duacs'):
            DUACS = True
        else:
            print('No DUACS-related parameters --> no diagnostics will be computed')
            DUACS = False
        print('\n* Read interpolated fields')
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
            
        data = read_data(comp.path_out+exp.name_experiment, opts.prods, ncentred, time_offset, DUACS=DUACS)
        
        #+++++++++++++++++++++++++++++++#
        #    Spectral Analysis          #
        #+++++++++++++++++++++++++++++++#
        print('\n* Spectral Analysis')
        spec = ana_spec(data, opts.prods, DUACS=DUACS)
        
        #+++++++++++++++++++++++++++++++#
        #    RMSE Analysis              #
        #+++++++++++++++++++++++++++++++#
        print('\n* RMSE Analysis')
        rmse = ana_rmse(data, opts.prods, DUACS=DUACS)
        
        #+++++++++++++++++++++++++++++++#
        #    Dump Analysis              #
        #+++++++++++++++++++++++++++++++#
        print('\n* Dump Analysis')        
        with open(file_outputs, 'wb') as f:
                pickle.dump((spec, rmse), f)
        



    
    
    
    
