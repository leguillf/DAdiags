#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:37:20 2019

@author: leguillou
"""

import sys,os
import numpy as np 
import calendar
from scipy import interpolate
from scipy.interpolate import griddata
import argparse
import load
import switchvar
import netCDF4 as nc
from datetime import datetime
import xarray as xr


def _datetime2timestamps(datetimes):
    timestamps = np.zeros_like(datetimes)
    for i,dt in enumerate(datetimes):
        timestamps[i] = calendar.timegm(dt.timetuple())
    return timestamps


def interpTime(datetimes_in,datetimes_out,fields_in): 
    # Convert to timestamps
    timestamps_in = _datetime2timestamps(datetimes_in)
    timestamps_out = _datetime2timestamps(datetimes_out)    
    # Time interpolation 
    f = interpolate.interp1d(timestamps_in, fields_in, axis=0)
    # Initialisation
    fields_out = np.empty((timestamps_out.size,fields_in.shape[1],fields_in.shape[2]))
    for i,t in enumerate(timestamps_out):
        if t<=timestamps_in.min():
            fields_out[i] = fields_in[0]
            print("Warning: A target timestamp is below the interpolation range, we set the field as the first one")
        elif t>=timestamps_in.max():
            fields_out[i] = fields_in[-1]
            print("Warning: A target timestamp is above the interpolation range, we set the field as the last one")
        else:
            fields_out[i] = f(t)

    return fields_out

def interpGrid(grid_in, grid_out, fields_in,interp='cubic'):

    """
    NAME 
        interpGrid

    DESCRIPTION
        
        Args:     
        grid_in (lon,lat)
        grid_out (lon,lat) 
        fields_in
    
        Returns: 
        fields_out

    """
    if (np.all(grid_in[0] == grid_out[0])) and (np.all(grid_in[1] == grid_out[1])) : 
        print("The grid are identical, no need to interpolate")
        return fields_in
    
    # Get lon,lat
    lon_in,lat_in = grid_in
    lon_out,lat_out = grid_out
    # Get output dimensions
    NY, NX = lon_out.shape
    NT = fields_in.shape[0]
    # Initialisation
    fields_out = np.empty((NT,NY,NX))
    # 2D space interpolation 
    for t in range(NT):
        fields_out[t,:,:] = griddata((lon_in.ravel(),lat_in.ravel()), fields_in[t].ravel(), (lon_out.ravel(),lat_out.ravel()), method=interp).reshape((NY,NX))
    
    return fields_out
        

def writeOutputs(path_out,prods_ref,prods_duacs,prods_da,names_prods,datetimes,lon,lat):
    
    if os.path.exists(path_out) is False:
        os.makedirs(path_out)
        
    varout = {}
    coords = {}
    coords['lon'] = (('y','x',),lon)
    coords['lat'] = (('y','x',),lat)
    coords['time'] = (('time',),[np.datetime64(dt) for dt in datetimes])

    for i,name in enumerate(names_prods):
        # Ref
        varout[name+'_ref'] = (('time','y','x',),prods_ref[i])  
        # DUACS
        if prods_duacs is not None:
            varout[name+'_duacs'] = (('time','y','x',), prods_duacs[i])  
        # DA
        varout[name+'_da'] = (('time','y','x',),prods_da[i])  
    
    ds = xr.Dataset(varout,coords=coords)
    ds.to_netcdf(path_out+'/data_interpolated.nc')

    return 

##======================================================================================================================##
##                MAIN                                                                                                  ##
##======================================================================================================================##

if __name__ == '__main__':
    #+++++++++++++++++++++++++++++++#
    #    _#
    #+++++++++++++++++++++++++++++++#
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path_config_exp', default=None, type=str)     # parameters relative to the DA experiment
    parser.add_argument('--path_config_comp', default=None, type=str)    # parameters relative to NATL60 and DUACS 
    parser.add_argument('--prods', default=['ssh'],nargs='+', type=str)
    parser.add_argument('--path_save', default=None, type=str)           # Path where the outputs are saved
    opts = parser.parse_args()
    
    #+++++++++++++++++++++++++++++++#
    #    GET params                 #
    #+++++++++++++++++++++++++++++++#
    print('\n* Get parameters')
    # parameters relative to the DA experiment
    dir_exp = os.path.dirname(opts.path_config_exp)
    name_exp = os.path.basename(opts.path_config_exp)
    if name_exp[-3:]=='.py':
        name_exp = name_exp[:-3]
    sys.path.insert(0,dir_exp)
    exp = __import__(name_exp, globals=globals())
    print(name_exp)
    if opts.path_save is not None:
        path_save = opts.path_save
    else:
        path_save = exp.path_save
    # parameters relative to NATL60 and DUACS 
    dir_comp = os.path.dirname(opts.path_config_comp)
    name_comp = os.path.basename(opts.path_config_comp)
    if name_comp[-3:]=='.py':
        name_comp = name_comp[:-3]
    sys.path.insert(0,dir_comp)
    comp = __import__(name_comp, globals=globals())

    #+++++++++++++++++++++++++++++++#
    #    Load products              #
    #+++++++++++++++++++++++++++++++#
    print('\n* Load data')            
    # Reference
    print('\t Reference')
    ssh_ref,datetime_ref,lon_ref,lat_ref = load.load_dataset(
            comp.path_reference,comp.file_reference,comp.name_time_reference,comp.name_lon_reference,comp.name_lat_reference,comp.name_var_reference,
            comp.time_min,comp.time_max,comp.lon_min,comp.lon_max,comp.lat_min,comp.lat_max,comp.options_ref,comp.dtout)
    
    # DA experiment
    print('\t DA Exp')
    ssh_da, datetime_da, lon_da, lat_da = load.load_dataset(
            exp.path_save,'/*.nc','time',exp.name_mod_lon,exp.name_mod_lat,exp.name_mod_var[0],
            comp.time_min,comp.time_max,comp.lon_min,comp.lon_max,comp.lat_min,comp.lat_max,comp.options_exp,comp.dtout)
    
    # DUACS
    if hasattr(comp, 'path_duacs'):
        DUACS = True
    else:
        print('No DUACS-related parameters --> no diagnostics will be computed')
        DUACS = False
    if DUACS:
        print('DUACS')
        ssh_duacs, datetime_duacs, lon_duacs, lat_duacs = load.load_DUACSprods(
                comp.path_duacs,comp.file_duacs,comp.name_time_duacs,comp.name_lon_duacs,comp.name_lat_duacs,comp.name_var_duacs,bounds=[lon_ref.min(),lon_ref.max(),lat_ref.min(),lat_ref.max()])

    #+++++++++++++++++++++++++++++++#
    #    Switch var                 #
    #+++++++++++++++++++++++++++++++#   
    print("\n* Switch variables : ", ','.join(opts.prods))
    if hasattr(exp, 'c0'):
        c = exp.c0
    else:
        print('Warning: argument "c" is not defined in experiment config file. Its value is set to 2.2')
        c = 2.2
    # Reference
    print('Reference')
    prods_ref = switchvar.ssh2multiple(ssh_ref,lon_ref,lat_ref,opts.prods,c,name_grd='grid_'+exp.name_experiment)
    # DUACS
    if DUACS:
        print('DUACS')
        prods_duacs = switchvar.ssh2multiple(ssh_duacs,lon_duacs,lat_duacs,opts.prods,c,name_grd='grid_'+exp.name_experiment)
    # DA
    print('DA')
    prods_da = switchvar.ssh2multiple(ssh_da,lon_da,lat_da,opts.prods,c,name_grd='grid_'+exp.name_experiment)

    #+++++++++++++++++++++++++++++++#
    #    Time interpolation         #
    #+++++++++++++++++++++++++++++++#    
    print('\n* Time interpolation')
    # DUACS
    if DUACS:
        print('DUACS')
        prods_duacs = [interpTime(datetime_duacs,datetime_ref,prod) for prod in prods_duacs]
    # DA
    print('DA')
    prods_da = [interpTime(datetime_da,datetime_ref,prod) for prod in prods_da]
    
    #+++++++++++++++++++++++++++++++#
    #    Grid interpolation         #
    #+++++++++++++++++++++++++++++++#
    print('\n* Grid interpolation')
    # DUACS
    if DUACS:
        print('DUACS')
        prods_duacs = [interpGrid((lon_duacs,lat_duacs),(lon_ref,lat_ref),prod) for prod in prods_duacs]
    else:
        prods_duacs = None
    # DA
    print('DA')
    prods_da = [interpGrid((lon_da,lat_da),(lon_ref,lat_ref),prod) for prod in prods_da]
    
    #+++++++++++++++++++++++++++++++#
    #    Write outputs              #
    #+++++++++++++++++++++++++++++++#
    print('\n* Write outputs')
    writeOutputs(comp.path_out+exp.name_experiment,prods_ref,prods_duacs,prods_da,opts.prods,datetime_ref,lon_ref,lat_ref)
    

    
    
    
    
    
    
