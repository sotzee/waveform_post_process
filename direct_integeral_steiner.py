#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 14:47:19 2018

@author: sotzee
"""

import numpy as np
from eos_class import EOS_BPS#,EOS_BPSwithPoly

baryon_density_s= 0.16
baryon_density0=0.16/2.7
baryon_density1=1.85*0.16
baryon_density2=3.7*0.16
baryon_density3=7.4*0.16
Preset_Pressure_final=1e-6
pressure0=EOS_BPS.eosPressure_frombaryon(baryon_density0)
density0=EOS_BPS.eosDensity(pressure0)
Preset_pressure_center_low=10
Preset_Pressure_final_index=1

std_Lambda_ratio=0.04

def Density_i(pressure_i,baryon_density_i,pressure_i_minus,baryon_density_i_minus,density_i_minus):
    gamma_i=np.log(pressure_i/pressure_i_minus)/np.log(baryon_density_i/baryon_density_i_minus)
    return gamma_i,(density_i_minus-pressure_i_minus/(gamma_i-1))*\
            (pressure_i/pressure_i_minus)**(1./gamma_i)+pressure_i/(gamma_i-1)

def p3_max(pressure1,pressure2):
    density1=Density_i(pressure1,baryon_density1,pressure0,baryon_density0,density0)[1]
    density2=Density_i(pressure2,baryon_density2,pressure1,baryon_density1,density1)[1]
    gamma3_max=1+density2/pressure2
    return pressure2*(baryon_density3/baryon_density2)**gamma3_max

import cPickle
dir_name='Lambda_hadronic_calculation'
f_file=open('./'+dir_name+'/Lambda_hadronic_calculation_p1p2p3_eos.dat','rb')
p1p2p3,eos=np.array(cPickle.load(f_file))
f_file.close()
shape=np.shape(p1p2p3)[0:3]+(-1,)

f_maxmass_result='./'+dir_name+'/Lambda_hadronic_calculation_maxmass.dat'
f_file=open(f_maxmass_result,'rb')
maxmass_result=np.reshape(cPickle.load(f_file),shape)
f_file.close()

f_mass_beta_Lambda_result = './'+dir_name+'/Lambda_hadronic_calculation_mass_beta_Lambda.dat'
f_file=open(f_mass_beta_Lambda_result,'rb')
mass_beta_Lambda_result=np.array(cPickle.load(f_file))
f_file.close()
mass=np.reshape(mass_beta_Lambda_result[:,0],shape)
beta=np.reshape(mass_beta_Lambda_result[:,1],shape)
Lambda=np.reshape(mass_beta_Lambda_result[:,2],shape)

from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
p1_range=(p1p2p3[:,0,0,0].min()[0],p1p2p3[:,0,0,0].max()[0])
p2_range=(100,'causal maximum')
p3_range=('np.max([1.2*p2,250])','p3_max')

f_p2_causal = './'+dir_name+'/Lambda_hadronic_calculation_p2_causal.dat'
f_file=open(f_p2_causal,'rb')
p2_causal_p1,p2_causal_p2=np.array(cPickle.load(f_file))
f_file.close()
causality_p2_int=interp1d(p2_causal_p1,p2_causal_p2)
#plt.plot(np.linspace(*p1_range),p2_causal,'.')
#plt.plot(np.linspace(*(p1_range)+(200,)),causality_p2_int(np.linspace(*(p1_range)+(200,))))

maxmass_int = RegularGridInterpolator((range(shape[0]), range(shape[1]), range(shape[2])), maxmass_result[:,:,:,1])
pc_max_int  = RegularGridInterpolator((range(shape[0]), range(shape[1]), range(shape[2])), maxmass_result[:,:,:,0])
cs2_max_int = RegularGridInterpolator((range(shape[0]), range(shape[1]), range(shape[2])), maxmass_result[:,:,:,2])

def get_p1p2p3_i(p1,p2,p3):
    p1_i=(p1-p1_range[0])/(p1_range[1]-p1_range[0])*(shape[0]-1)
    p2_max=causality_p2_int(p1)
    p2_i=(p2-p2_range[0])/(p2_max-p2_range[0])*(shape[1]-1)
    log_p3_min=np.log(np.max([1.2*p2,250]))
    log_p3_max=np.log(p3_max(p1,p2))
    p3_i=(np.log(p3)-log_p3_min)/(log_p3_max-log_p3_min)*(shape[2]-1)
    return [p1_i,p2_i,p3_i]

def get_maxmass(p1,p2,p3):
    p1p2p3_i=get_p1p2p3_i(p1,p2,p3)
    return pc_max_int(p1p2p3_i),maxmass_int(p1p2p3_i),cs2_max_int(p1p2p3_i)

star_N=np.shape(mass)[-1]
mass_int  = RegularGridInterpolator((range(shape[0]), range(shape[1]), range(shape[2]), range(star_N)), mass)
log_tidal_int = RegularGridInterpolator((range(shape[0]), range(shape[1]), range(shape[2]), range(star_N)), np.log(Lambda))

f_log_Lambda_mass_grid_result='./'+dir_name+'/Lambda_hadronic_calculation_log_Lambda_mass_grid.dat'
#from Parallel_process import main_parallel
#main_parallel(Calculation_log_Lambda_mass_grid,mass_beta_Lambda_result,f_log_Lambda_mass_grid_result,0)
f_file=open(f_log_Lambda_mass_grid_result,'rb')
log_Lambda_mass_grid=np.reshape(np.array(cPickle.load(f_file)),shape)
f_file.close()
mass_grid=np.linspace(1.0,2.0,37)
log_Lambda_mass_grid_int = RegularGridInterpolator((range(shape[0]), range(shape[1]), range(shape[2]), mass_grid), log_Lambda_mass_grid)



redshift=0.00903738

import h5py
filename = 'uniform_mass_prior_common_eos_20hz_lowfreq_posteriors.hdf'
fp = h5py.File(filename, "r")
print(fp.attrs['variable_args'])
m1 = (fp['samples/mass1'][:100]/(1+redshift)).flatten()
m2 = (fp['samples/mass2'][:100]/(1+redshift)).flatten()
Lambdasym = (fp['samples/lambdasym'][:100]).flatten()
q = m2/m1
fp.close()
Lambda1=Lambdasym*(m2/m1)**3
Lambda2=Lambdasym*(m1/m2)**3
array_mass=np.concatenate((m1,m2))
array_log10_Lambda=np.log10(np.concatenate((Lambda1,Lambda2)))
from scipy.stats import gaussian_kde
kernel = gaussian_kde(np.vstack([array_mass, array_log10_Lambda]))

import matplotlib.pyplot as plt
m_plot=np.linspace(1,1.8,81)
log10_Lambda_plot=np.linspace(1,3.2,67)
x,y=np.meshgrid(m_plot,log10_Lambda_plot)
plt.figure(figsize=(10,8))
plt.imshow((kernel([x.flatten(),y.flatten()]).reshape(np.shape(x))),aspect = 'auto',origin='lower',extent=[np.min(m_plot),np.max(m_plot),np.min(log10_Lambda_plot),np.max(log10_Lambda_plot)])
plt.plot(array_mass,array_log10_Lambda,'ro')

def rand_uniform(x_min,x_max,N):
    return np.random.rand(N)*(x_max-x_min)+x_min
N=100
m_array=rand_uniform(1.0, 1.8,N)
p1_array=rand_uniform(3.75, 30,N)
p2_array=rand_uniform(100, 200,N)
p3_array=rand_uniform(300, 1000,N)


def Lambda(mass,p1,p2,p3):
    if(p2>causality_p2_int(p1)):
        return 0*mass+1
    pc_max,maxmass,cs2_max=get_maxmass(p1,p2,p3)
    if(maxmass<2 or cs2_max>1):
        return 0*mass+1
    else:
        p1p2p3_i=get_p1p2p3_i(p1,p2,p3)
        tidal_list=[]
        for mass_i in mass:
            tidal_list.append(log_Lambda_mass_grid_int(p1p2p3_i+[mass_i])[0])
        return np.exp(tidal_list)
