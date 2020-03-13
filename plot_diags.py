#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 13:01:54 2020

@author: leguillou
"""

import os, sys
import argparse 
import pickle
import matplotlib.gridspec as gridspec
import matplotlib.pylab as plt
import numpy as np
from analysis import ana_compute_rmse



def diag_multiple_DAfields_comp(refField,expField,lon2d,lat2d,dt_curr,RMSE=None,itime=0,name_RefFields='True state',name_DAfields=None,prefixe='',var='SSH',name_var=None,ncentred=None,var_range=None,cmap='RdBu_r',save=False,path_save=None,plot_err_from_ref=False, xlabel='time (days)',ylabel='RMSE',gdtr_obs=None):
    
    import matplotlib.pyplot as plt
    
    Nexp,NY,NX = expField.shape
    if var_range is None:
        if ncentred is not None:
            min_ = expField[:,ncentred:-(ncentred+1),ncentred:-(ncentred+1)].min()
            max_ = expField[:,ncentred:-(ncentred+1),ncentred:-(ncentred+1)].max()
        else:
            min_ = expField.min()        
            max_ = expField.max()
    else:
        min_ = var_range[0]
        max_ = var_range[1]
        
    # Create 2xNda sub plots
    if RMSE is not None:
        gs = gridspec.GridSpec(2, Nexp+2,width_ratios=[1,]*(Nexp+1) + [0.1])
        fig = plt.figure(figsize=(Nexp*10, 5*Nexp)) 
    else:
        gs = gridspec.GridSpec(1, Nexp+2,width_ratios=[1,]*(Nexp+1) + [0.1])
        fig = plt.figure(figsize=(Nexp*10, int(2.5*Nexp)) )
    
    fig.suptitle(dt_curr.strftime("%Y-%m-%d"),fontsize=20)
    
    # Compute error from reference if this option is activated
    if plot_err_from_ref:
        expField = expField - refField
      
    # Plot Reference
    ax0 = plt.subplot(gs[0, 0])
    ax0.pcolormesh(lon2d, lat2d, refField,cmap=plt.cm.get_cmap(cmap),vmin=min_,vmax=max_)
    ax0.set_title(name_RefFields,fontsize=20)
    
    # Plot obs groundtracks
    if gdtr_obs is not None:
        ax0.scatter(gdtr_obs['lon'],gdtr_obs['lat'],c=gdtr_obs['c'],s=gdtr_obs['s'],alpha=0.3)
        ax0.set_xlim(lon2d.min(),lon2d.max())
        ax0.set_ylim(lat2d.min(),lat2d.max())
        
    # Plot Da fields
    for iexp in range(Nexp):
        ax = plt.subplot(gs[0, iexp+1])
        im = ax.pcolormesh(lon2d, lat2d, expField[iexp],cmap=plt.cm.get_cmap(cmap),vmin=min_,vmax=max_)
        if name_DAfields is not None:
            ax.set_title(name_DAfields[iexp],fontsize=20)
        else:
            ax.set_title('DA product nb ' +str(iexp),fontsize=20)

         # Plot subarea where RMSE is computed 
        if ncentred is not None:            
            ax.plot([lon2d[ncentred,ncentred],lon2d[ncentred,NX-ncentred-1],lon2d[NY-ncentred-1,NX-ncentred-1],lon2d[NY-ncentred-1,ncentred],lon2d[ncentred,ncentred]],
                    [lat2d[ncentred,ncentred],lat2d[ncentred,NX-ncentred-1],lat2d[NY-ncentred-1,NX-ncentred-1],lat2d[NY-ncentred-1,ncentred],lat2d[ncentred,ncentred]],'-k')    
    
    
    # Colorbar
    cax = plt.subplot(gs[0, -1])
    cbar = fig.colorbar(im, cax=cax, format='%.0e')
    cbar.ax.set_ylabel(var,fontsize=15)
    
    if RMSE is not None:
        Nt = RMSE.shape[1]
        # Plot RMSE 
        ax1 = plt.subplot(gs[1, :])
        for iexp in range(Nexp):
            rmse_t = np.zeros_like(RMSE[0])
            rmse_t[:itime] = RMSE[iexp,:itime]
            rmse_t[itime:] = np.nan
            ax1.plot(rmse_t,label=name_DAfields[iexp])
        ax1.set_xlabel(xlabel,fontsize=15)
        ax1.set_ylabel(ylabel,fontsize=15)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
        ax1.legend(fontsize=15)
        ax1.set_ylim([np.nanmin(RMSE),np.nanmax(RMSE)])
        ax1.set_xlim([0,Nt])
        
    # Save figure
    if save:        
        if name_var is not None:
            string = name_var
        else:
            string = var
        title = "comparison_" + string + '_' + prefixe + '_' + dt_curr.strftime("%Y-%m-%d-%H%M")+'.png'
        plt.savefig(path_save + title)
        plt.axis('off')        
        plt.close()
        return title
    else:
        plt.show()   
        return 
    

    



def diag_multiple_DAfields_comp_main(refFields,expFields,lon2d,lat2d,timein,name_RefFields='True state',name_DAfields=None,prefixe='',var='SSH',name_var=None,ncentred=None,var_range=None,plot_rmse=False,cmap='RdBu_r',save=False,path_save=None,plot_err_from_ref=False,run_parallel=False, xlabel='time (days)',ylabel='RMSE',dict_obs=None):
    """
    NAME 
        diag_multiple_DAfields_comp_main

    DESCRIPTION
    

        Args:     
        
            
        Param:
        

        Returns: 
        

    """
    
    
    if path_save is not None and not os.path.exists(path_save):
        os.makedirs(path_save)
     
    Nt = timein.size
    Nexp = expFields.shape[0]
    
    if var_range is None:
        var_range = [0,0]
        if ncentred is not None:
            var_range[0] = expFields[:,:,ncentred:-(ncentred+1),ncentred:-(ncentred+1)].min()
            var_range[1] = expFields[:,:,ncentred:-(ncentred+1),ncentred:-(ncentred+1)].max()
        else:
            var_range[0] = expFields.min()        
            var_range[1] = expFields.max()
        
    if plot_rmse:
        RMSE = []
        for i in range(Nexp):
            RMSE.append(ana_compute_rmse(refFields, expFields[i], ncentred=ncentred))
        RMSE = np.asarray(RMSE)
    else:
        RMSE = None
    
    if run_parallel:
        from dask import delayed
        ## Defining temporary function for dask parallelization
        def process(iii): 
            diag_multiple_DAfields_comp(
                path_save=path_save,refField=refFields[iii,:,:],expField=expFields[:,iii,:,:],lon2d=lon2d,lat2d=lat2d,itime=iii,dt_curr=timein[iii],RMSE=RMSE,
                name_RefFields=name_RefFields,name_DAfields=name_DAfields,prefixe=prefixe,var=var,ncentred=ncentred,var_range=var_range,cmap=cmap,save=save,plot_err_from_ref=plot_err_from_ref,
                xlabel=xlabel,ylabel=ylabel)
            return 1
 
        # Run diag in parallel on the timeserie
        output = [] 
        for i in range(Nt):
            outpng=delayed(process)(i) 
            output.append(outpng)   
        total = delayed(print)(output)      
        total.compute() 
        
    else:
        for i,dt in enumerate(timein):
            if dict_obs is not None and dt in dict_obs:
                gdtr_obs = dict_obs[dt]
            else: 
                gdtr_obs = None
                
            out = diag_multiple_DAfields_comp(
                    path_save=path_save,refField=refFields[i,:,:],expField=expFields[:,i,:,:],lon2d=lon2d,lat2d=lat2d,itime=i,dt_curr=dt,RMSE=RMSE,
                    name_RefFields=name_RefFields,name_DAfields=name_DAfields,prefixe=prefixe,var=var,name_var=name_var,ncentred=ncentred,var_range=var_range,cmap=cmap,save=save,plot_err_from_ref=plot_err_from_ref,
                    xlabel=xlabel,ylabel=ylabel,
                    gdtr_obs=gdtr_obs)
            print(out)
    return



def plot_1d(ana_rmse, ana_spec, name_var, labels, path_save):
    # Plot parameters          
    cm = plt.get_cmap('CMRmap') 
    
    nprod = len(ana_rmse) 
    
    colors = [cm(i/nprod) for i in range(nprod)]
    linestyles=["-","--","-.",":"]*3
    
    padx = 0.06
    pady = 0.06

    fig = plt.figure(figsize=(30,20))
    
    ########################################## 
    #                 RMSE                   #
    ########################################## 
    print('RMSE')
    #outer
    outergs = gridspec.GridSpec(1, 1)
    outergs.update(bottom=0.5, left=0, right = 1,
                   top=1)
    outerax = fig.add_subplot(outergs[0])
    outerax.tick_params(axis='both',which='both',bottom=0,left=0,
                        labelbottom=0, labelleft=0)
    #outerax.set_facecolor("gold")
    plt.setp(outerax.spines.values(), linewidth=5)
    
    #inner
    gs = gridspec.GridSpec(1, 2)
    gs.update(bottom=0.5+pady, left=padx, right=1-padx/4,
                   top=1-pady)
    axe_RMSE = fig.add_subplot(gs[0])
    axe_RMSE_S = fig.add_subplot(gs[1])
    for i in range(len(labels)):        
        ## DA
        num_rmse = str(round(ana_rmse[i]['Num']['da'][name_var],2))
        axe_RMSE.plot(ana_rmse[i]['RMSE']['da'][name_var],color=colors[i],linestyle=linestyles[i],label=labels[i])
        axe_RMSE_S.plot(ana_rmse[i]['Score']['da'][name_var],color=colors[i],linestyle=linestyles[i],label=num_rmse)
    
    # Axe configuration
    axe_RMSE.set_title('RMSE')
    axe_RMSE.set_ylabel('m')
    axe_RMSE_S.set_title('RMSE-based score')
    axe_RMSE.set_xlabel('Time (days)')
    axe_RMSE_S.set_xlabel('Time (days)')
    axe_RMSE.set_ylim(ymin=0)
    axe_RMSE_S.set_ylim(0,1)
    axe_RMSE.ticklabel_format(style='sci', axis='y',scilimits=(0,0))  
    
    # Legends
    axe_RMSE.legend()
    leg = axe_RMSE_S.legend(handlelength=0, handletextpad=0, fancybox=True)
    leg.set_title( 'Mean score:')
    for i,(item,txt) in enumerate(zip(leg.legendHandles,leg.texts)):
        item.set_visible(False)
        txt.set_color(colors[i])
        
    ########################################## 
    #                 Spec                   #
    ########################################## 
    print('Spec')
    #outer
    outergs = gridspec.GridSpec(1, 1)
    outergs.update(bottom=0, left=0, right = 1,
                   top=0.5)
    outerax = fig.add_subplot(outergs[0])
    outerax.tick_params(axis='both',which='both',bottom=0,left=0,
                        labelbottom=0, labelleft=0)
    #outerax.set_facecolor("indigo")
    plt.setp(outerax.spines.values(), linewidth=5)
    
    #inner
    gs = gridspec.GridSpec(1, 2)
    gs.update(bottom=pady, left=padx, right = 1-padx/4,
                   top=0.5-pady)
    outerax.patch.set_alpha(0.3)
    outerax.tick_params(width=2)
    axe_SPEC = fig.add_subplot(gs[0])
    axe_SPEC_S = fig.add_subplot(gs[1])
    for i in range(len(labels)):        
        ## DA
        axe_SPEC.loglog(1e3*ana_spec[i]['wavenumber'],ana_spec[i]['PSD']['ref'][name_var],color=colors[i],linewidth=2*params['lines.linewidth'],label='NR ' + labels[i])
        num_spec = str(round(ana_spec[i]['Num']['da'][name_var]*1e-3,1)) + ' km'
        axe_SPEC.loglog(1e3*ana_spec[i]['wavenumber'],ana_spec[i]['PSD']['da'][name_var],color=colors[i],linestyle=linestyles[i],label='DA ' + labels[i])    
        axe_SPEC_S.semilogx(1e3*ana_spec[i]['wavenumber'],ana_spec[i]['Score']['da'][name_var],color=colors[i],linestyle=linestyles[i],label=num_spec)
        axe_SPEC_S.axvline(x=1e3/ana_spec[i]['Num']['da'][name_var],color=colors[i],ymax=0.5,linewidth=2,linestyle='--')
    
    # Axe configuration 
    axe_SPEC.set_title('wavenumber PSD')
    axe_SPEC.set_ylabel(r'$m^2$')
    axe_SPEC_S.set_title('PSD-based score')
    axe_SPEC_S.set_ylim(0,1)
    axe_SPEC.grid(True,which="both",ls="--")
    axe_SPEC_S.xaxis.grid()
    axe_SPEC.set_xlabel('wavenumber (cpkm)')
    axe_SPEC_S.set_xlabel('wavenumber (cpkm)')
    
    # Legends
    axe_SPEC.legend()
    leg = axe_SPEC_S.legend(handlelength=0, handletextpad=0, fancybox=True)
    leg.set_title( 'Effective resolution:')
    for i,(item,txt) in enumerate(zip(leg.legendHandles,leg.texts)):
        item.set_visible(False)
        txt.set_color(colors[i])
        
    fig.savefig(path_save)
    
    
    
    
    
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
    # optional parameters that have to be provided if *path_config_exp* is not provided
    parser.add_argument('--name_exp', default=None, type=str)               
    # optional parameters that have to be provided if *name_config_comp* is not provided
    parser.add_argument('--path_out', default=None, type=str)   
    
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
        else:
            print('Error: either name_config_comp or path_out has to be specified')
            sys.exit()            
    else:           
        sys.path.insert(0,os.path.join(os.path.dirname(__file__), "configs"))
        comp = __import__(opts.name_config_comp, globals=globals())
        path_out = comp.path_out
        
    #+++++++++++++++++++++++++++++++#
    #    Plot                       #
    #+++++++++++++++++++++++++++++++#
    print('\n* Plot')
    params = {
                'font.size'           : 8      ,
                'axes.labelsize': 23,
                'axes.titlesize': 30,
                'xtick.labelsize'     : 20      ,
                'ytick.labelsize'     : 20      ,
                'legend.fontsize': 25,
                'legend.handlelength': 2,
                'lines.linewidth':4,
                'legend.title_fontsize':25}
    plt.rcParams.update(params)
    
    file_ana = path_out+name_exp +'/analysis.pic'
    with open(file_ana, 'rb') as f:
        rmse, spec, wk = pickle.load(f)
    
    # Plot 1d    
    for prod in opts.prods:
        path_save = path_out+name_exp +'/plot_1d_' + prod
        plot_1d([rmse], [spec], prod, [name_exp], path_save)