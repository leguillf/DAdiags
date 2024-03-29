#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 14:59:19 2019

@author: leguillou
"""

import argparse 
import sys, os
import numpy as np 
from scipy import interpolate
import pickle

def compute_mean_rmse(rmse, prods, name):
    
    mean_rmse = [] 
    rmse_data = rmse['Score']
    for prod in prods:
        mean_rmse.append(np.mean(rmse_data[name][prod]))
        
    return mean_rmse

def compute_space_res(wk, prods, name, r=0.5):
    
    space_res = []
    wavenumber = wk['wavenumber']   # in cycle/km
    
    for prod in prods:
        score = 1 - wk['Err'][name][prod].mean(axis=0)/wk['WK']['ref'][prod].mean(axis=0)
        f = interpolate.interp1d(score, 1/wavenumber, axis=0)   
        try:
            res = f(r)
        except:
            res = np.nan
        space_res.append(res)
        
    return space_res

def compute_time_res(wk, prods, name, r=0.5):

    time_res = []
    frequency = wk['frequency']  # in cycle/day
    for prod in prods:
        score = 1 - wk['Err'][name][prod].mean(axis=1)/wk['WK']['ref'][prod].mean(axis=1)
        f = interpolate.interp1d(score, 1/frequency, axis=0)
        try:
            res = f(r)
        except:
            res = np.nan
        time_res.append(res)

    return time_res

def write_outputs(file,mean_rmse_duacs,mean_rmse_da,space_res_duacs,space_res_da,time_res_duacs,time_res_da, prods):
    f = open(file,'w')
    # DUACS
    if mean_rmse_duacs is not None and space_res_duacs is not None and time_res_duacs is not None:
        f.write('DUACS\n')
        f.write('\t RMSE:\n' )
        for i,prod in enumerate(prods):
            f.write('\t\t' + prod + ': ' + f"{(mean_rmse_duacs[i]):02d}" + '\n') 
        f.write('\t Space Res:\n' )
        for i,prod in enumerate(prods):
            f.write('\t\t' + prod + ': ' + f"{(space_res_duacs[i]):02d}" + '\n')
        f.write('\t Time Res:\n' )
        for i,prod in enumerate(prods):
            f.write('\t\t' + prod + ': ' + f"{(time_res_duacs[i]):02d}" + '\n')
    # DA
    f.write('\nDA\n')
    f.write('\t RMSE:\n' )
    for i,prod in enumerate(prods):
        f.write('\t\t' + prod + ': ' + f"{(mean_rmse_da[i]):02f}" + '\n') 
    f.write('\t Space Res:\n' )
    for i,prod in enumerate(prods):
        f.write('\t\t' + prod + ': ' + f"{(space_res_da[i]):02f}" + '\n')
    f.write('\t Time Res:\n' )
    for i,prod in enumerate(prods):
        f.write('\t\t' + prod + ': ' + f"{(time_res_da[i]):02f}" + '\n')
                

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
    if name_exp[-3:]=='.py':
        name_exp = name_exp[:-3]
    exp = __import__(name_exp, globals=globals())
    # parameters relative to NATL60 and DUACS
    dir_comp = os.path.dirname(opts.path_config_comp)
    name_comp = os.path.basename(opts.path_config_comp)
    sys.path.insert(0,dir_comp)
    if name_comp[-3:]=='.py':
        name_comp = name_comp[:-3]
    comp = __import__(name_comp, globals=globals())
    
    #+++++++++++++++++++++++++++++++#
    #    Diagnostics                #
    #+++++++++++++++++++++++++++++++#
    file_inputs = comp.path_out+exp.name_experiment +'/analysis.pic'
    file_outputs = comp.path_out+exp.name_experiment +'/diags.txt'
    if os.path.isfile(file_outputs) and opts.overwrite!=1: 
        print(file_outputs, 'already exists')
    else:
        if opts.overwrite==1: 
            print('Warning:', file_outputs, 'already exists but you ask to overwrite it')
        # DUACS
        if hasattr(comp, 'path_duacs'):
            DUACS = True
        else:
            print('No DUACS-related parameters --> no diagnostics will be computed')
            DUACS = False
        #+++++++++++++++++++++++++++++++#
        #    Read analysis              #
        #+++++++++++++++++++++++++++++++#
        print('\n* Read Analysis')        
        with open(file_inputs, 'rb') as f:
            rmse,spec,wk = pickle.load(f)
        
        #+++++++++++++++++++++++++++++++#
        #    Evaluate metrics           #
        #+++++++++++++++++++++++++++++++#
        print('\n* Evaluate metrics ')
        mean_rmse_da = compute_mean_rmse(rmse, comp.prods, name='da')
        space_res_da = compute_space_res(wk, comp.prods, name='da')
        print(space_res_da)
        time_res_da = compute_time_res(wk, comp.prods, name='da')
        print(time_res_da)
        if DUACS:
            mean_rmse_duacs = compute_mean_rmse(rmse, comp.prods, name='duacs')
            space_res_duacs = compute_space_res(wk, comp.prods, name='duacs')
            time_res_duacs = compute_time_res(wk, comp.prods, name='da')
        else:
            mean_rmse_duacs = space_res_duacs = time_res_duacs = None
            
        #+++++++++++++++++++++++++++++++#
        #    Write outputs              #
        #+++++++++++++++++++++++++++++++#
        print('\n* Write outputs ')
        print(file_outputs)
        write_outputs(file_outputs, mean_rmse_duacs,mean_rmse_da,space_res_duacs,space_res_da,time_res_duacs,time_res_da, comp.prods)
        

        
        
