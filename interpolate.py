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
    if (np.all(grid_in == grid_out)):
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
        
    ncout = nc.Dataset(path_out + '/data_interpolated.nc', 'w', format='NETCDF3_CLASSIC')
    print(path_out + '/data_interpolated.nc')
    # Dimensions
    ncout.createDimension('time', datetimes.size)
    ncout.createDimension('x', lon.shape[1])
    ncout.createDimension('y', lat.shape[0]) 
    # Grid & timestamps
    nctime = ncout.createVariable('time', 'f', ('time',))
    nclon = ncout.createVariable('lon', 'f', ('y','x',))
    nclat = ncout.createVariable('lat', 'f', ('y','x',))
    nctime[:] = _datetime2timestamps(datetimes)
    nclon[:,:] = lon
    nclat[:,:] = lat
    for i,name in enumerate(names_prods):
        # Ref
        ncref = ncout.createVariable(name+'_ref', 'f', ('time','y','x',))  
        ncref[:,:,:] = prods_ref[i]
        # DUACS
        ncduacs = ncout.createVariable(name+'_duacs', 'f', ('time','y','x',))  
        ncduacs[:,:,:] = prods_duacs[i]
        # DA
        ncda = ncout.createVariable(name+'_da', 'f', ('time','y','x',))  
        ncda[:,:,:] = prods_da[i]
    # Close file
    ncout.close()
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
    parser.add_argument('--name_config_comp', default=None, type=str)    # parameters relative to NATL60 and DUACS 
    parser.add_argument('--prods', default=['ssh'],nargs='+', type=str)
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
    print(name_exp)
    # parameters relative to NATL60 and DUACS 
    sys.path.insert(0,os.path.join(os.path.dirname(__file__), "configs"))
    comp = __import__(opts.name_config_comp, globals=globals())
    print(opts.name_config_comp)

    #+++++++++++++++++++++++++++++++#
    #    Load products              #
    #+++++++++++++++++++++++++++++++#
    print('\n* Load products')            
    # NATL60
    print('NATL60')
    ssh_ref,datetime_ref,lon_ref,lat_ref = load.load_natl60ssh(
            comp.path_reference,comp.file_reference,comp.name_time_reference,comp.name_lon_reference,comp.name_lat_reference,comp.name_var_reference,
            exp.init_date-exp.propagation_time_step,exp.final_date+exp.propagation_time_step,datetime_type=True)
    # DUACS
    print('DUACS')
    ssh_duacs, datetime_duacs, lon_duacs, lat_duacs = load.load_DUACSprods(
            comp.path_duacs,comp.file_duacs,comp.name_time_duacs,comp.name_lon_duacs,comp.name_lat_duacs,comp.name_var_duacs,bounds=[lon_ref.min(),lon_ref.max(),lat_ref.min(),lat_ref.max()])
    # DA experiment
    print('DA')
    ssh_da, datetime_da, lon_da, lat_da = load.load_DAprods(exp.path_save,exp.name_mod_lon,exp.name_mod_lat,exp.name_mod_var[0],exp.init_date,exp.final_date,exp.propagation_time_step,prefixe=exp.name_exp_save)
    
    #+++++++++++++++++++++++++++++++#
    #    Switch var                 #
    #+++++++++++++++++++++++++++++++#   
    print("\n* Switch variables : ", ','.join(opts.prods))
    if hasattr(exp, 'c'):
        c = exp.c
    else:
        print('Warning: argument "c" is not defined in experiment config file. Its value is set to 2.2')
        c = 2.2
    # NATL60
    print('NATL60')
    prods_ref = switchvar.ssh2multiple(ssh_ref,lon_ref,lat_ref,opts.prods,c)
    # DUACS
    print('DUACS')
    prods_duacs = switchvar.ssh2multiple(ssh_duacs,lon_duacs,lat_duacs,opts.prods,c)
    # DA
    print('DA')
    prods_da = switchvar.ssh2multiple(ssh_da,lon_da,lat_da,opts.prods,c)

    #+++++++++++++++++++++++++++++++#
    #    Time interpolation         #
    #+++++++++++++++++++++++++++++++#    
    print('\n* Time interpolation')
    # DUACS
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
    print('DUACS')
    prods_duacs = [interpGrid((lon_duacs,lat_duacs),(lon_ref,lat_ref),prod) for prod in prods_duacs]
    # DA
    print('DA')
    prods_da = [interpGrid((lon_da,lat_da),(lon_ref,lat_ref),prod) for prod in prods_da]
    
    #+++++++++++++++++++++++++++++++#
    #    Write outputs              #
    #+++++++++++++++++++++++++++++++#
    print('\n* Write outputs')
    writeOutputs(comp.path_out+exp.name_experiment,prods_ref,prods_duacs,prods_da,opts.prods,datetime_ref,lon_ref,lat_ref)
    

    
    
    
    
    
    
