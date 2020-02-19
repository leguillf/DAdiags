#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 14:42:13 2019

@author: leguillou
"""

# path and netcdf file of the NATL60 outputs
path_reference = '/mnt/meom/workdir/leguilfl/DATA/NATL60/regular_grid/daily/'
file_reference = 'NATL60GULFSTREAM_2012-10-01_2013-09-30.1d.nc' #'NATL60OSMOSIS_2012-10-01_2013-09-10.1d.nc'
path_duacs = '/mnt/meom/workdir/leguilfl/DATA/DUACS-OI_maps/ssh_model/'
file_duacs = 'ssh_sla_boost_NATL60_swot.nc'

# Name of the time and grid variables in the NATL60 netcdf file.
name_time_reference = 'time'#'time'
name_lon_reference  = 'nav_lon'
name_lat_reference  = 'nav_lat'
name_var_reference  = 'sossheig'#'sossheig'

# Name of the grid variables in the DUACS product netcdf files.
name_time_duacs = 'time'
name_lon_duacs =  'lon'
name_lat_duacs = 'lat'
name_var_duacs = 'ssh'

ncentred = 60

time_offset = 30

interp = 'cubic'

path_out =  '/mnt/meom/workdir/leguilfl/experiences/DUACS_comparisons/'
