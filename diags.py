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
    rmse_data = rmse['RMSE']
    for prod in prods:
        mean_rmse.append(np.mean(rmse_data[name][prod]))
        
    return mean_rmse

def compute_eff_res(spec, prods, name, r=0.5):
    
    eff_res = []
    wavenumber = spec['wavenumber'] *1e3  # in cycle/km
    recScore_data = spec['recScore']
    for prod in prods:
        f = interpolate.interp1d(recScore_data[name][prod], 1/wavenumber, axis=0)   
        eff_res.append(f(r))
        
    return eff_res

def write_outputs(file,mean_rmse_duacs,mean_rmse_da,eff_res_duacs,eff_res_da, prods):
    f = open(file,'w')
    # DUACS
    if mean_rmse_duacs is not None and eff_res_duacs is not None:
        f.write('DUACS\n')
        f.write('\t RMSE:\n' )
        for i,prod in enumerate(prods):
            f.write('\t\t' + prod + ': ' + "{:.3E}".format(mean_rmse_duacs[i]) + '\n') 
        f.write('\t Eff Res:\n' )
        for i,prod in enumerate(prods):
            f.write('\t\t' + prod + ': ' + "{:.3E}".format(eff_res_duacs[i]) + '\n')            
    # DA
    f.write('\nDA\n')
    f.write('\t RMSE:\n' )
    for i,prod in enumerate(prods):
        f.write('\t\t' + prod + ': ' + "{:.3E}".format(mean_rmse_da[i]) + '\n') 
    f.write('\t Eff Res:\n' )
    for i,prod in enumerate(prods):
        f.write('\t\t' + prod + ': ' + "{:.3E}".format(eff_res_da[i]) + '\n') 
                

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
            spec, rmse = pickle.load(f)
        
        #+++++++++++++++++++++++++++++++#
        #    Evaluate metrics           #
        #+++++++++++++++++++++++++++++++#
        print('\n* Evaluate metrics ')
        mean_rmse_da = compute_mean_rmse(rmse, opts.prods, name='da')
        eff_res_da = compute_eff_res(spec, opts.prods, name='da')
        if DUACS:
            mean_rmse_duacs = compute_mean_rmse(rmse, opts.prods, name='duacs')
            eff_res_duacs = compute_eff_res(spec, opts.prods, name='duacs')
        else:
            mean_rmse_duacs = eff_res_duacs = None
        
        #+++++++++++++++++++++++++++++++#
        #    Write outputs              #
        #+++++++++++++++++++++++++++++++#
        print('\n* Write outputs ')
        print(file_outputs)
        write_outputs(file_outputs, mean_rmse_duacs,mean_rmse_da,eff_res_duacs,eff_res_da, opts.prods)
        

        
        
