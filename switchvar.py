#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:59:26 2019

@author: leguillou
"""


import numpy as np 
import pickle
import os                                  
from math import cos, pi 

def ssh2multiple(ssh,lon,lat,varnames,c=2.5,name_grd=None):
    var_list = []
    for var in varnames:
        if var=='ssh':
            var_list.append(ssh)
        elif var=='vel':
            u,v = ssh2uv(ssh,lon,lat,name_grd=name_grd)
            var_list.append(np.sqrt(u**2+v**2))
        elif var=='rv':
            rv = ssh2rv(ssh,lon,lat,name_grd=name_grd)
            var_list.append(rv)
        elif var=='pv':
            pv = ssh2pv(ssh,lon,lat,c,name_grd=name_grd)
            var_list.append(pv)
        else:
            print(var, 'is not recognized')
            var_list.append(ssh*np.nan)
            
    os.system("rm "+name_grd +'_switchvar')       
    
    return var_list
        


def ssh2pv(ssh,lon,lat,c,name_grd=None,xac=None):
    if name_grd is not None:
        # Grid
        name_grd += '_switchvar'
        if os.path.isfile(name_grd):   
            with open(name_grd, 'rb') as f:
               grd = pickle.load(f)
        else:
            grd = grid(lon,lat)
            with open(name_grd, 'wb') as f:
               pickle.dump(grd, f)
               f.close()
    else: 
        grd = grid(lon,lat)   

    f0 = grd.f0
    g = grd.g 
    # Initialization
    pv = np.zeros_like(ssh)    
    if len(ssh.shape) == 2: # One map
        # Compute relative vorticity
        #pv = factor*laplacian(ssh,dx,dy) - g*f0/(c**2) * ssh 
        pv[1:-1,1:-1] = g/grd.f0[1:-1,1:-1]*((ssh[2:,1:-1]+ssh[:-2,1:-1]-2*ssh[1:-1,1:-1])/grd.dy[1:-1,1:-1]**2 \
                                      + (ssh[1:-1,2:]+ssh[1:-1,:-2]-2*ssh[1:-1,1:-1])/grd.dx[1:-1,1:-1]**2) \
                     - g*f0[1:-1,1:-1]/(c**2) * ssh[1:-1,1:-1] 
        if xac is not None:
            pv = _masked_edge(pv,xac)       
    elif len(ssh.shape) ==3: # Time serie of maps (or multiple layer)
        NT = ssh.shape[0]
        for t in range(NT):            
            ssh_t = ssh[t]
            # Compute relative vorticity
            #pv[t] = laplacian(factor*ssh_t,dx,dy) - g*f0/(c**2) * ssh_t 
            pv[t,1:-1,1:-1] = g/grd.f0[1:-1,1:-1]*((ssh_t[2:,1:-1]+ssh_t[:-2,1:-1]-2*ssh_t[1:-1,1:-1])/grd.dy[1:-1,1:-1]**2 \
                                      + (ssh_t[1:-1,2:]+ssh_t[1:-1,:-2]-2*ssh_t[1:-1,1:-1])/grd.dx[1:-1,1:-1]**2) \
                     - g*f0[1:-1,1:-1]/(c**2) * ssh_t[1:-1,1:-1] 
            if xac is not None:
                pv[t] = _masked_edge(pv[t],xac)   
                
    return pv  

def ssh2uv(ssh,lon,lat,name_grd=None,xac=None):
    if name_grd is not None:
        # Grid
        name_grd += '_switchvar'
        if os.path.isfile(name_grd):   
            with open(name_grd, 'rb') as f:
               grd = pickle.load(f)
        else:
            grd = grid(lon,lat)
            with open(name_grd, 'wb') as f:
               pickle.dump(grd, f)
               f.close()
    else: 
        grd = grid(lon,lat)
    g = grd.g    

    # Initialization
    u = np.zeros_like(ssh)    
    v = np.zeros_like(ssh)    
    if len(ssh.shape) == 2: # One map
        # velocities
        u[1:-1,1:-1] = -0.5*g/grd.f0[1:-1,1:-1]*(ssh[2:,1:-1] - ssh[:-2,1:-1]) / grd.dy[1:-1,1:-1]
        v[1:-1,1:-1] = 0.5*g/grd.f0[1:-1,1:-1]*(ssh[1:-1,2:] - ssh[1:-1,:-2]) / grd.dx[1:-1,1:-1]
        if xac is not None:
            u = _masked_edge(u,xac)       
            v = _masked_edge(v,xac)       
    elif len(ssh.shape) ==3: # Time serie of maps (or multiple layer)
        NT = ssh.shape[0]
        for t in range(NT):            
            ssh_t = ssh[t]
            # Compute vorticities
            u[t,1:-1,1:-1] = -0.5*g/grd.f0[1:-1,1:-1]*(ssh_t[2:,1:-1] - ssh_t[:-2,1:-1]) / grd.dy[1:-1,1:-1]
            v[t,1:-1,1:-1] = 0.5*g/grd.f0[1:-1,1:-1]*(ssh_t[1:-1,2:] - ssh_t[1:-1,:-2]) / grd.dx[1:-1,1:-1]
            if xac is not None:
                u[t] = _masked_edge(u[t],xac)           
                v[t] = _masked_edge(v[t],xac)  

    return u,v  
    
def ssh2rv(ssh,lon,lat,name_grd=None,xac=None):
    if name_grd is not None:
        # Grid
        name_grd += '_switchvar'
        if os.path.isfile(name_grd):   
            with open(name_grd, 'rb') as f:
               grd = pickle.load(f)
        else:
            grd = grid(lon,lat)
            with open(name_grd, 'wb') as f:
               pickle.dump(grd, f)
               f.close()
    else: 
        grd = grid(lon,lat)
        
    g = grd.g

    # Initialization
    rv = np.zeros_like(ssh)    
    if len(ssh.shape) == 2: # One map
        # Compute relative vorticity
        #rv = laplacian(factor*ssh,dx,dy) #- ((g/f0**2))*beta*gradj(ssh)/dy   
        rv[1:-1,1:-1]=g/grd.f0[1:-1,1:-1]*((ssh[2:,1:-1]+ssh[:-2,1:-1]-2*ssh[1:-1,1:-1])/grd.dy[1:-1,1:-1]**2 \
                                      + (ssh[1:-1,2:]+ssh[1:-1,:-2]-2*ssh[1:-1,1:-1])/grd.dx[1:-1,1:-1]**2) 
        if xac is not None:
            rv = _masked_edge(rv,xac)       
    elif len(ssh.shape) ==3: # Time serie of maps (or multiple layer)
        NT = ssh.shape[0]
        for t in range(NT):            
            ssh_t = ssh[t]
            # Compute relative vorticity
            #rv[t] = laplacian(factor*ssh_t,dx,dy) #- ((g/f0**2))*beta*gradj(ssh_t)/dy
            rv[t,1:-1,1:-1]=g/grd.f0[1:-1,1:-1]*((ssh_t[2:,1:-1]+ssh_t[:-2,1:-1]-2*ssh_t[1:-1,1:-1])/grd.dy[1:-1,1:-1]**2 \
                                      + (ssh_t[1:-1,2:]+ssh_t[1:-1,:-2]-2*ssh_t[1:-1,1:-1])/grd.dx[1:-1,1:-1]**2) 
            if xac is not None:
                rv[t] = _masked_edge(rv[t],xac)        
    return rv  


def uv2rv(UV,lon,lat,name_grd=None,xac=None):
    if name_grd is not None:
        # Grid
        name_grd += '_switchvar'
        if os.path.isfile(name_grd):   
            with open(name_grd, 'rb') as f:
               grd = pickle.load(f)
        else:
            grd = grid(lon,lat)
            with open(name_grd, 'wb') as f:
               pickle.dump(grd, f)
               f.close()
    else: 
        grd = grid(lon,lat)
    dx = grd.dx
    dy = grd.dy
        
    # Initialization
    u = UV[0]
    mask = np.isnan(u)
    v = UV[1]
    rv = np.zeros_like(u)
    if len(u.shape) == 2: # One map
        rv = _gradj(v)/dx - _gradi(u)/dy 
        if xac is not None:
            rv = _masked_edge(rv,xac)
    elif len(u.shape) ==3: # Time serie of maps
        NT = u.shape[0]
        for t in range(NT):            
            u_t = u[t]
            v_t = v[t]
            # Compute relative vorticity
            rv[t] = _gradj(v_t)/dx - _gradi(u_t)/dy 
            if xac is not None:
                rv[t] = _masked_edge(rv[t],xac)
    rv[mask] = np.nan
    
    return rv 

def pv2ssh(lon,lat,q,hg,c,nitr=1,name_grd=''):
    """ Q to SSH
    
    This code solve a linear system of equations using Conjugate Gradient method

    Args:
        q (2D array): Potential Vorticity field
        hg (2D array): SSH guess
        grd (Grid() object): check modgrid.py

    Returns:
        h (2D array): SSH field. 
    """
    def compute_avec(vec,aaa,bbb,grd):
    
        avec=np.empty(grd.np0,) 
        avec[grd.vp2]=aaa[grd.vp2]*((vec[grd.vp2e]+vec[grd.vp2w]-2*vec[grd.vp2])/(grd.dx1d[grd.vp2]**2)+(vec[grd.vp2n]+vec[grd.vp2s]-2*vec[grd.vp2])/(grd.dy1d[grd.vp2]**2)) + bbb[grd.vp2]*vec[grd.vp2]
        avec[grd.vp1]=vec[grd.vp1]
     
        return avec,
    if name_grd is not None:
        # Grid
        name_grd += '_switchvar_pv2ssh'
        if os.path.isfile(name_grd):   
            with open(name_grd, 'rb') as f:
               grd = pickle.load(f)
        else:
            grd = grid(lon,lat,c=c)
            with open(name_grd, 'wb') as f:
               pickle.dump(grd, f)
               f.close()
    else: 
        grd = grid(lon,lat)        
        
    ny,nx,=np.shape(hg)
    g=grd.g


    x=hg[grd.indi,grd.indj]
    q1d=q[grd.indi,grd.indj]

    aaa=g/grd.f01d
    bbb=-g*grd.f01d/grd.c1d**2
    ccc=+q1d

    aaa[grd.vp1]=0
    bbb[grd.vp1]=1
    ccc[grd.vp1]=x[grd.vp1]  ##boundary condition

    vec=+x

    avec,=compute_avec(vec,aaa,bbb,grd)
    gg=avec-ccc
    p=-gg

    for itr in range(nitr-1): 
        vec=+p
        avec,=compute_avec(vec,aaa,bbb,grd)
        tmp=np.dot(p,avec)
        
        if tmp!=0. : s=-np.dot(p,gg)/tmp
        else: s=1.
        
        a1=np.dot(gg,gg)
        x=x+s*p
        vec=+x
        avec,=compute_avec(vec,aaa,bbb,grd)
        gg=avec-ccc
        a2=np.dot(gg,gg)
        
        if a1!=0: beta=a2/a1
        else: beta=1.
        
        p=-gg+beta*p
    
    vec=+p
    avec,=compute_avec(vec,aaa,bbb,grd)
    val1=-np.dot(p,gg)
    val2=np.dot(p,avec)
    if (val2==0.): 
        s=1.
    else: 
        s=val1/val2

    a1=np.dot(gg,gg)
    x=x+s*p

    # back to 2D
    h=np.empty((ny,nx))
    h[:,:]=np.NAN
    h[grd.indi,grd.indj]=x[:]


    return h



class grid():

  def __init__(self,lon,lat, c=None):
    
    ny,nx,=np.shape(lon)
    if c is None:
        c = 2.7

    mask=np.zeros((ny,nx))+2
    mask[:2,:]=1
    mask[:,:2]=1
    mask[-3:,:]=1
    mask[:,-3:]=1
    dx=np.zeros((ny,nx))
    dy=np.zeros((ny,nx))

    for i in range(1,ny-1):
      for j in range(1,nx-1):
        dlony=lon[i+1,j]-lon[i,j]
        dlaty=lat[i+1,j]-lat[i,j]
        dlonx=lon[i,j+1]-lon[i,j]
        dlatx=lat[i,j+1]-lat[i,j]
        dx[i,j]=np.sqrt((dlonx*111000*cos(lat[i,j]*pi/180))**2 + (dlatx*111000)**2)
        dy[i,j]=np.sqrt((dlony*111000*cos(lat[i,j]*pi/180))**2 + (dlaty*111000)**2)
    
    dx[0,:]=dx[1,:]
    dx[-1,:]=dx[-2,:] 
    dx[:,0]=dx[:,1]
    dx[:,-1]=dx[:,-2]
    dy[0,:]=dy[1,:]
    dy[-1,:]=dy[-2,:] 
    dy[:,0]=dy[:,1]
    dy[:,-1]=dy[:,-2]

    f0=2*2*pi/86164*np.sin(lat*pi/180) 

    np0=np.shape(np.where(mask>=1))[1]
    np2=np.shape(np.where(mask==2))[1]
    np1=np.shape(np.where(mask==1))[1]
    self.mask1d=np.zeros((np0))
    self.H=np.zeros((np0))
    self.c1d=np.zeros((np0))
    self.f01d=np.zeros((np0))
    self.dx1d=np.zeros((np0))
    self.dy1d=np.zeros((np0))
    self.indi=np.zeros((np0), dtype=np.int)
    self.indj=np.zeros((np0), dtype=np.int)
    self.vp1=np.zeros((np1), dtype=np.int)
    self.vp2=np.zeros((np2), dtype=np.int)
    self.vp2=np.zeros((np2), dtype=np.int)
    self.vp2n=np.zeros((np2), dtype=np.int)
    self.vp2nn=np.zeros((np2), dtype=np.int)
    self.vp2s=np.zeros((np2), dtype=np.int)
    self.vp2ss=np.zeros((np2), dtype=np.int)
    self.vp2e=np.zeros((np2), dtype=np.int)
    self.vp2ee=np.zeros((np2), dtype=np.int)
    self.vp2w=np.zeros((np2), dtype=np.int)
    self.vp2ww=np.zeros((np2), dtype=np.int)
    self.vp2nw=np.zeros((np2), dtype=np.int)
    self.vp2ne=np.zeros((np2), dtype=np.int)
    self.vp2se=np.zeros((np2), dtype=np.int)
    self.vp2sw=np.zeros((np2), dtype=np.int)
    self.indp=np.zeros((ny,nx), dtype=np.int) 

    p=-1
    for i in range(ny):
      for j in range(nx):
        if (mask[i,j]>=1):
          p=p+1
          self.mask1d[p]=mask[i,j]
          self.dx1d[p]=dx[i,j]
          self.dy1d[p]=dy[i,j]
          self.f01d[p]=f0[i,j]
          self.indi[p]=i
          self.indj[p]=j
          self.indp[i,j]=p
          self.c1d[p]=c
 
 
    p2=-1
    p1=-1
    for p in range(np0):
      if (self.mask1d[p]==2):
        p2=p2+1
        i=self.indi[p]
        j=self.indj[p]
        self.vp2[p2]=p
        self.vp2n[p2]=self.indp[i+1,j]
        self.vp2nn[p2]=self.indp[i+2,j]
        self.vp2s[p2]=self.indp[i-1,j]
        self.vp2ss[p2]=self.indp[i-2,j]
        self.vp2e[p2]=self.indp[i,j+1]
        self.vp2ee[p2]=self.indp[i,j+2]
        self.vp2w[p2]=self.indp[i,j-1]
        self.vp2ww[p2]=self.indp[i,j-2]
        self.vp2nw[p2]=self.indp[i+1,j-1]
        self.vp2ne[p2]=self.indp[i+1,j+1]
        self.vp2se[p2]=self.indp[i-1,j+1]
        self.vp2sw[p2]=self.indp[i-1,j-1]
      if (self.mask1d[p]==1):
        p1=p1+1
        i=self.indi[p]
        j=self.indj[p]
        self.vp1[p1]=p
    self.mask=mask
    self.f0=f0
    self.dx=dx
    self.dy=dy
    self.np0=np0
    self.np2=np2
    self.c=c
    self.nx=nx
    self.ny=ny 
    self.lon=lon
    self.lat=lat
    self.g=9.81
    
    
def _masked_edge(var,xac):
    ind_edge_swath = np.transpose((xac==np.nanmin(xac)) | (xac==np.nanmax(xac)))
    var[ind_edge_swath] = np.nan
    if np.any(xac>0) and np.any(xac<0):
        ind_edge_swath_gap = np.transpose((xac==np.nanmin(xac[xac>0])) | (xac==np.nanmax(xac[xac<0])))
    elif np.any(xac>0):
        ind_edge_swath_gap = np.transpose((xac==np.nanmin(xac[xac>0])))
    else:
        ind_edge_swath_gap = np.transpose((xac==np.nanmax(xac[xac<0])))
    var[ind_edge_swath_gap] = np.nan
        
    return var

def _gradi(I): 
    """
    Calculates the gradient in the x-direction of an image I and gives as output M.
    In order to keep the size of the initial image the last row is left as 0s.
    """
    
    m, n = I.shape
    M = np.zeros([m,n])*np.nan

    M[0:-1,:] = np.subtract(I[1::,:], I[0:-1,:])
    return M

def _gradj(I): 
    """
    Calculates the gradient in the y-direction of an image I and gives as output M.
    In order to keep the size of the initial image the last column is left as 0s.
    """
    
    m, n = I.shape
    M = np.zeros([m,n])
    M[:,0:-1] =  np.subtract(I[:,1::], I[:,0:-1])
    return M
