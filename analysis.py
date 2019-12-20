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

def read_data(path_data, prods, bc):
    
    data = {}
    ncin = nc.Dataset(path_data+'/data_interpolated.nc')
    # Time and grid
    data['time'] = ncin.variables['time'][:]
    data['lon'] = ncin.variables['lon'][bc:-(bc+1),bc:-(bc+1)]
    data['lat'] = ncin.variables['lat'][bc:-(bc+1),bc:-(bc+1)]
    # Variables
    data['ref'] = {}
    data['duacs'] = {}
    data['da'] = {}
    for prod in prods:
        data['ref'][prod] = ncin.variables[prod+'_ref'][:,bc:-(bc+1),bc:-(bc+1)]
        data['duacs'][prod] = ncin.variables[prod+'_duacs'][:,bc:-(bc+1),bc:-(bc+1)]
        data['da'][prod] = ncin.variables[prod+'_da'][:,bc:-(bc+1),bc:-(bc+1)]
    
    return data



def ana_spec(data, prods):
    
    PSD = {'ref':{},'duacs':{},'da':{}}
    recScore = {'duacs':{},'da':{}}
    
    NT = data['time'].size
    lon = data['lon']
    lat = data['lat']
    
    # Loop on variables 
    for prod in prods:
        # Initialize lists
        PSD['ref'][prod] = []
        PSD['duacs'][prod] = []
        PSD['da'][prod] = []
        recScore['duacs'][prod] = []
        recScore['da'][prod] = []
        # Time loop
        for t in range(NT):
            # Compute PSD of the fields and the erros at each timestamp
            wavenumber, psd2D_ref_t = wavenumber_spectra(data['ref'][prod][t],lon,lat) 
            wavenumber, psd2D_duacs_t = wavenumber_spectra(data['duacs'][prod][t],lon,lat) 
            wavenumber, psd2D_da_t = wavenumber_spectra(data['da'][prod][t],lon,lat) 
            wavenumber, psd2D_err_duacs_t = wavenumber_spectra(data['duacs'][prod][t]-data['ref'][prod][t],lon,lat) 
            wavenumber, psd2D_err_da_t = wavenumber_spectra(data['da'][prod][t]-data['ref'][prod][t],lon,lat)            
            # Append to list
            PSD['ref'][prod].append(psd2D_ref_t)
            PSD['duacs'][prod].append(psd2D_duacs_t)
            PSD['da'][prod].append(psd2D_da_t)
            recScore['duacs'][prod].append(psd2D_err_duacs_t)
            recScore['da'][prod].append(psd2D_err_da_t)                
        # Average temporally 
        PSD['ref'][prod] = np.mean(PSD['ref'][prod],axis=0)
        PSD['duacs'][prod] = np.mean(PSD['duacs'][prod],axis=0)
        PSD['da'][prod] = np.mean(PSD['da'][prod],axis=0)
        recScore['duacs'][prod] = 1 - np.mean(recScore['duacs'][prod],axis=0)/np.mean(PSD['ref'][prod],axis=0)
        recScore['da'][prod] = 1 - np.mean(recScore['da'][prod],axis=0)/np.mean(PSD['ref'][prod],axis=0)

    return {'wavenumber':wavenumber,'PSD':PSD, 'recScore':recScore}




def ana_rmse(data, prods):
    
    RMSE = {'duacs':{},'da':{}}
    
    for prod in prods:  
        NT,NY,NX = data['ref'][prod].shape          
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
        print('\n* Read interpolated fields')
        data = read_data(comp.path_out+exp.name_experiment, opts.prods, exp.lenght_bc)
        
        #+++++++++++++++++++++++++++++++#
        #    Spectral Analysis          #
        #+++++++++++++++++++++++++++++++#
        print('\n* Spectral Analysis')
        spec = ana_spec(data, opts.prods)
        
        #+++++++++++++++++++++++++++++++#
        #    RMSE Analysis              #
        #+++++++++++++++++++++++++++++++#
        print('\n* RMSE Analysis')
        rmse = ana_rmse(data, opts.prods)
        
        #+++++++++++++++++++++++++++++++#
        #    Dump Analysis              #
        #+++++++++++++++++++++++++++++++#
        print('\n* Dump Analysis')        
        with open(file_outputs, 'wb') as f:
                pickle.dump((spec, rmse), f)
        



    
    
    
    
