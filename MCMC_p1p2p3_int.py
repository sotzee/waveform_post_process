#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 22:21:35 2018

@author: sotzee
"""

import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
from eos_class import EOS_BPS,EOS_BPSwithPoly

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


# =============================================================================
# def Lambda(mass,p1,p2,p3):
#     eos=EOS_BPSwithPoly([baryon_density0,p1,baryon_density1,p2,baryon_density2,p3,baryon_density3])
#     maxmass_result=Maxmass(Preset_Pressure_final,Preset_rtol,eos)
#     args=[eos,maxmass_result[1],maxmass_result[2]]
#     Lambda_list=[]
#     for mass_i in mass:
#         if(mass_i>args[2]):
#             Lambda_list.append(0)
#         else:
#             ofmass_result=Properity_ofmass(mass_i,Preset_pressure_center_low,args[1],MassRadius,Preset_Pressure_final,Preset_rtol,Preset_Pressure_final_index,args[0])
#             Lambda_list.append(ofmass_result[5])
#     return Lambda_list
# =============================================================================

from pycbc import cosmology
distance=40.7 # in Mpc
redshift=cosmology.redshift(distance)

import h5py
filename = 'uniform_mass_prior_common_eos_20hz_lowfreq_posteriors.hdf'
fp = h5py.File(filename, "r")
print fp.attrs['variable_args']
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
p1_dist=pm.Uniform("p1", 3.75, 30)
p2_dist=pm.Uniform("p2", 100, 200)
p3_dist=pm.Uniform("p3", 300, 1000)
@pm.deterministic
def center_i(array_mass=array_mass,p1_dist=p1_dist,p2_dist=p2_dist,p3_dist=p3_dist):
    return np.log10(Lambda(array_mass,p1_dist,p2_dist,p3_dist))

center_i_result=center_i
observations = pm.Normal("obs", center_i_result, 1/(std_Lambda_ratio*center_i_result)**2, value=array_log10_Lambda, observed=True)

model = pm.Model([observations, p1_dist,p2_dist,p3_dist])

mcmc = pm.MCMC(model)



def Lambda(mass,p1,p2,p3):
    if(p2>causality_p2_int(p1)):
        return 0*mass+1
    pc_max,maxmass,cs2_max=get_maxmass(p1,p2,p3)
    if(maxmass<2 or cs2_max>1):
        return 0*mass+1
    else:
        p1p2p3_i=get_p1p2p3_i(p1,p2,p3)
        return np.exp(log_Lambda_mass_grid_int(p1p2p3_i+[mass])[0])
m_dist=pm.Uniform("mass", 1.0, 1.8)
p1_dist=pm.Uniform("p1", 3.75, 30)
p2_dist=pm.Uniform("p2", 100, 200)
p3_dist=pm.Uniform("p3", 300, 1000)

@pm.stochastic(observed=True)
def custom_stochastic(value=m_dist=m_dist,p1_dist=p1_dist,p2_dist=p2_dist,p3_dist=p3_dist):
    return kernel([m_dist,Lambda(m_dist,p1_dist,p2_dist,p3_dist)])

model = pm.Model([ p1_dist,p2_dist,p3_dist,custom_stochastic])

mcmc = pm.MCMC(model)


from cPickle import dump,load
for i in range(3):
    mcmc.sample(5000)
    file_data=open('./p1p2p3_trace_ratio_%.2f.dat%d'%(std_Lambda_ratio,i+1),'wb')
    dump(np.array([mcmc.trace("p1")[:],mcmc.trace("p2")[:],mcmc.trace("p3")[:]]),file_data)
    file_data.close()

traces=[]
for i in range(3):
    file_data=open('./p1p2p3_trace_ratio_%.2f.dat%d'%(std_Lambda_ratio,i+1),'rb')
    traces.append(load(file_data))
    file_data.close()

sample_i=1
plt.figure(figsize=(12.5, 9))
lw = 1
p1p2p3_median=[]
for i in range(3):
    plt.subplot(3,1,i+1)
    p1p2p3_median.append(np.median(np.concatenate((traces[sample_i][i],traces[sample_i+1][i]))))
    plt.plot(np.concatenate((traces[sample_i][i],traces[sample_i+1][i])), label="p_1", lw=lw)

plt.figure(figsize=(12.5, 9))
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.hist(np.concatenate((traces[sample_i][i],traces[sample_i+1][i])), bins=30,
              histtype="stepfilled")
plt.xlabel('p$_{i=1,2,3}$ (MeV fm$^{-3})$',size=20)

plt.figure(figsize=(12.5, 9))
p1_post=np.concatenate((traces[sample_i][0],traces[sample_i+1][0]))
gamma1=np.log(p1_post/pressure0)/np.log(baryon_density1/baryon_density0)
L_post=3*pressure0/baryon_density0*(baryon_density_s/baryon_density0)**(gamma1-1)
plt.hist(L_post, bins=30,histtype="stepfilled")
plt.xlabel('L (MeV)',size=15)

plt.figure(figsize=(12.5, 9))
plt.plot(array_mass,array_log10_Lambda,'.',label='first 100 posterior points from De et. al.')
plt.plot(mass_grid,np.log10(Lambda(mass_grid,p1p2p3_median[0],p1p2p3_median[1],p1p2p3_median[2])),'.',label='eos with meadian p1,p2p3')
plt.legend()
plt.xlabel('mass/$\odot$',size=15)
plt.ylabel('log$_{10} \Lambda$',size=15)