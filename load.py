#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:22:18 2019

@author: leguillou
"""

import os, fnmatch
import netCDF4 as nc
import numpy as np 
from datetime import datetime,timedelta

def load_Refprods(directory,file,name_time,name_lon,name_lat,name_var,dt_start,dt_end,dt_ref=datetime(1958,1,1,0,0,0),datetime_type=True):
    """
    NAME 
        load_Refprods

    DESCRIPTION

        Args:  
            directory (string): directory in which the the netcdf file is stored
            file (string): name of the netcdf file
            name_time (string): name of time coordinate in the netcdf file
            name_lon (string): name of longitude coordinate in the netcdf files
            name_lat (string): name of latitude coordinate in the netcdf files
            name_var (string): name of the variable (typically ssh) to process in the netcdf files
            dt_start (datetime object): time of the begining of the analysis window 
            dt_end (datetime object): time of the end of the analysis window 
            dt_ref (datetime object): refrence time used in the netcdf file
            datetime_type (bool): if true, return datetype formatting timestamps
                
        Returns: 
            var (3D numpy array): fields of the simulation stored in a numpy array
            dt_out_time (1D numpy array): timestamps of the fields
            lon (2D numpy array): longitude of the fields
            lat (2D numpy array): latitude of the fields
    """
    
    # Compute time boundaries in seconds since dt_ref to be compared to timestamps
    time_sec_min = (dt_start - dt_ref).total_seconds()
    time_sec_max = (dt_end - dt_ref).total_seconds() 
    # Read timestamp and grid
    ncin = nc.Dataset(directory + file)
    timestamp = np.array(ncin.variables[name_time][:])  
    lon = np.array(ncin.variables[name_lon][:]) % 360
    lat = np.array(ncin.variables[name_lat][:]) 
    # Find time indexes corresponding to the time boundaries
    idx_time = (timestamp >= time_sec_min) & (timestamp <= time_sec_max)
    out_time = ncin.variables[name_time][idx_time]
    # Read variable 
    var = np.array(ncin.variables[name_var][idx_time,:,:]) 
    
    var[var<=-50] = 0.
    # Compute datetimes corresponding to the indexes found
    if datetime_type:
        dt_out_time = []
        for itime in range(len(out_time)):
            dt_out_time.append(dt_ref + timedelta(seconds=int(out_time[itime]))) 
    else: 
        dt_out_time = out_time
        
    return var, np.asarray(dt_out_time), lon, lat




def load_DAprods(directory, name_lon, name_lat, name_var, dt_start, dt_end, dt_timestep, prefixe = "*", suffixe = ""):
    """
    NAME 
        load_DAprods

    DESCRIPTION

        Args:  
            directory (string): directory in which the outputs are stored
            name_lon (string): name of longitude coordinate in the netcdf files
            name_lat (string): name of latitude coordinate in the netcdf files
            name_var (string): name of the variable (typically ssh) to process in the netcdf files
            dt_start (datetime object): time of the begining of the analysis window 
            dt_end (datetime object): time of the end of the analysis window 
            dt_timestep (timedelta object): timestep of the outputs (typically one hour)
            prefixe (string): specific prefix of the simulation, if several simulations are stored in the same directory
            suffixe (string): specific suffixe of the simulation, if several simulations are stored in the same directory
                
        Returns: 
            fields (3D numpy array): fields of the simulation stored in a numpy array
            datetimes (1D numpy array): timestamps of the fields
            lon (2D numpy array): longitude of the fields
            lat (2D numpy array): latitude of the fields
    """

    listOfFiles = os.listdir(directory)  
    fields = []
    datetimes = []
    dt_curr = dt_start
    first_iter = True
    while dt_curr <= dt_end :          
        yyyy_curr = str(dt_curr.year)
        mm_curr = str(dt_curr.month).zfill(2)
        dd_curr = str(dt_curr.day).zfill(2)
        HH_curr = str(dt_curr.hour).zfill(2)
        MM_curr = str(dt_curr.minute).zfill(2)
        pattern =  prefixe + '_y' + yyyy_curr + 'm' + mm_curr + 'd' + dd_curr + 'h' + HH_curr + MM_curr + suffixe + "*" + ".nc"
        file = [f for f in listOfFiles if fnmatch.fnmatch(f, pattern)]
        if len(file)>1:
            print("Error: several outputs match the pattern... Please set a prefixe.")
            return
        elif len(file)==0:
            print("No file matching the patter : " + pattern)
            dt_curr += dt_timestep
            continue
        else:
            file = file[0]
            fid_deg = nc.Dataset(directory + file) 
            if first_iter:
                lon = np.array(fid_deg.variables[name_lon][:,:]) % 360
                lat = np.array(fid_deg.variables[name_lat][:,:])  
                first_iter = False
            field_curr = np.mean(fid_deg.variables[name_var][:,:,:],axis=0)
            if np.any(np.isnan(field_curr)):
                print(file,' : NaN value found. Stop here !')
                break
            else:
                fields.append(field_curr)
                datetimes.append(dt_curr)
        dt_curr += dt_timestep
        
    return np.asarray(fields),np.asarray(datetimes),lon,lat


def load_DUACSprods(directory,file,name_time,name_lon,name_lat,name_ssh,dt_ref_duacs=datetime(1950,1,1,0,0,0),bounds=None):
    """
    NAME 
        load_DUACSprods

    DESCRIPTION

        Args:  
            directory (string): directory in which the outputs are stored
            file (string): name of the netcdf file
            name_time (string): name of time coordinate in the netcdf file
            name_lon (string): name of longitude coordinate in the netcdf file
            name_lat (string): name of latitude coordinate in the netcdf file
            name_ssh (string): name of ssh to process in the netcdf file
            dt_ref_duacs (datetime object): refrence time used in the netcdf file
            bounds (list): domain boundaries (optional) to extract the fields
           
        Returns: 
            ssh2d_duacs (3D numpy array): fields of the simulation stored in a numpy array
            datetimes_duacs (1D numpy array): timestamps of the fields
            lon2d_duacs (2D numpy array): longitude of the fields
            lat2d_duacs (2D numpy array): latitude of the fields
    """

    ncin = nc.Dataset(directory + file)
    time_duacs = np.array(ncin.variables[name_time][:]) 
    if 'swot_en_j1_tpn_g2' in file:
        # Time correction due to a bug in the data processing
        time_duacs += 22919 - 19358
    lon_duacs = np.array(ncin.variables[name_lon][:]) % 360
    lat_duacs = np.array(ncin.variables[name_lat][:]) 
    lon2d_duacs, lat2d_duacs = np.meshgrid(lon_duacs,lat_duacs)
    ssh2d_duacs = np.ma.array(ncin.variables[name_ssh][:,:,:]) 
    # Convert time to datetimes
    datetimes_duacs = [dt_ref_duacs + timedelta(days=d) for d in time_duacs]
    datetimes_duacs = np.asarray(datetimes_duacs)
    # Extraction
    if bounds is not None:
        lon_min,lon_max,lat_min,lat_max = bounds
        ind_lon = np.where(np.abs(lon_duacs -(lon_max+lon_min)/2) <= (lon_max-lon_min)/2 + 0.25)[0]
        ind_lat = np.where(np.abs(lat_duacs - (lat_max+lat_min)/2) <= (lat_max-lat_min)/2 + 0.25) [0]    
        lon2d_duacs = lon2d_duacs[ind_lat[0]:(ind_lat[-1]+1),ind_lon[0]:(ind_lon[-1]+1)]
        lat2d_duacs = lat2d_duacs[ind_lat[0]:(ind_lat[-1]+1),ind_lon[0]:(ind_lon[-1]+1)]
        ssh2d_duacs = ssh2d_duacs[:,ind_lat[0]:(ind_lat[-1]+1),ind_lon[0]:(ind_lon[-1]+1)]
        
    return ssh2d_duacs, datetimes_duacs, lon2d_duacs, lat2d_duacs
