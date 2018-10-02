# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

figsize_norm=1
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from pycbc import cosmology
from physicalconst import c,G,mass_sun
distance=40.7 # in Mpc
redshift=cosmology.redshift(distance)
print redshift

import h5py
filename = 'uniform_mass_prior_common_eos_20hz_lowfreq_posteriors.hdf'
fp = h5py.File(filename, "r")
print fp.attrs['variable_args']
m1 = (fp['samples/mass1'][:]/(1+redshift)).flatten()
m2 = (fp['samples/mass2'][:]/(1+redshift)).flatten()
Lambdasym = (fp['samples/lambdasym'][:]).flatten()
q = m2/m1
fp.close()
Lambda1=Lambdasym*(m2/m1)**3
Lambda2=Lambdasym*(m1/m2)**3
log10_Lambda1=np.log10(Lambda1)
log10_Lambda2=np.log10(Lambda2)

def tidal_binary(q,tidal1,tidal2):
    return 16.*((12*q+1)*tidal1+(12+q)*q**4*tidal2)/(13*(1+q)**5)
def mass_chirp(mass1,mass2):
    return (mass1*mass2)**0.6/(mass1+mass2)**0.2
Lambda_binary=tidal_binary(q,Lambda1,Lambda2)
m_chirp=mass_chirp(m1,m2)
log10_Lambda_binary=np.log10(Lambda_binary)

m=np.concatenate((m1,m2))
Lambda=np.concatenate((Lambda1,Lambda2))
log10_Lambda=np.log10(Lambda)
a=0.0085
radius1=m1*(Lambda1/a)**(1./6)*(mass_sun*G/c**2/100000)
radius2=m2*(Lambda2/a)**(1./6)*(mass_sun*G/c**2/100000)
radius_onepointfour=(radius2-radius1)*(1.4-m1)/(m2-m1)+radius1
radius=m*(Lambda/a)**(1./6)*(mass_sun*G/c**2/100000)
beta1=(mass_sun*G/c**2/100000)*m1/radius1
beta2=(mass_sun*G/c**2/100000)*m2/radius2
beta_m_chirp_radius_onepointfour=(mass_sun*G/c**2/100000)*m_chirp/radius_onepointfour
beta=(mass_sun*G/c**2/100000)*m/radius
Lambda_binary_beta_m_chirp_radius_onepointfour6_1000=1000*Lambda_binary*beta_m_chirp_radius_onepointfour**6

# =============================================================================
# a_lower=0.0075
# a_upper=0.0095
# radius_lower=m*(Lambda/a_upper)**(1./6)*(mass_sun*G/c**2/100000)
# radius_upper=m*(Lambda/a_lower)**(1./6)*(mass_sun*G/c**2/100000)
# a=[0.0075,0.0080,0.0085,0.0090,0.0095]
# m_extend=(m+0*np.array(a)[:,None]).flatten()
# radius_extend=m_extend*(Lambda/np.array(a)[:,None]).flatten()**(1./6)*(mass_sun*G/c**2/100000)
# =============================================================================
from collections import Counter
def get_density_2D(points_array,bin_num):
    points_array_x=points_array[0]
    points_array_y=points_array[1]
    bin_size_x=(points_array_x.max()-points_array_x.min())/(2*bin_num[0])
    bin_size_y=(points_array_y.max()-points_array_y.min())/(2*bin_num[1])
    x_lim = np.array([points_array_x.min(), points_array_x.max()])+0.5*bin_size_x
    y_lim = np.array([points_array_y.min(), points_array_y.max()])+0.5*bin_size_y
    xx, yy = np.mgrid[x_lim[0]:x_lim[1]:(bin_num[0]*1j), y_lim[0]:y_lim[1]:(bin_num[1]*1j)]
    
    points_array_x_int=((1./bin_size_x)*(points_array_x-points_array_x.min())).astype(int)
    points_array_y_int=((1./bin_size_y)*(points_array_y-points_array_y.min())).astype(int)
    points_array_x_int[points_array_x_int==bin_num[0]]=bin_num[0]-1
    points_array_y_int[points_array_y_int==bin_num[1]]=bin_num[1]-1
    points_array_int=np.array([points_array_x_int,points_array_y_int]).transpose()
    points_array_str=[str(a[0])+','+str(a[1]) for a in points_array_int]
    points_array_str_counts=Counter(points_array_str)
    counting_array=np.zeros(bin_num)
    for i in range(bin_num[0]):
        for j in range(bin_num[1]):
            counting_array[i,j]=points_array_str_counts['%d,%d'%(i,j)]
    return xx, yy, counting_array

from scipy.stats import gaussian_kde
def get_kde_2D(points_array,bin_num):
    points_array_x=points_array[0]
    points_array_y=points_array[1]
    bin_size_x=(points_array_x.max()-points_array_x.min())/(2*bin_num[0])
    bin_size_y=(points_array_y.max()-points_array_y.min())/(2*bin_num[1])
    x_lim = np.array([points_array_x.min(), points_array_x.max()])+0.5*bin_size_x
    y_lim = np.array([points_array_y.min(), points_array_y.max()])+0.5*bin_size_y
    xx, yy = np.mgrid[x_lim[0]:x_lim[1]:(bin_num[0]*1j), y_lim[0]:y_lim[1]:(bin_num[1]*1j)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([points_array_x, points_array_y])
    kernel = gaussian_kde(values)
    kde_array = np.reshape(kernel(positions).T, xx.shape)
    return xx, yy, kde_array

bin_num=[100,100]
m1m2_array=get_density_2D([m1,m2],bin_num)
radius_array,m_array,radius_m_array=get_density_2D([radius,m],bin_num)

log10_Lambda_array,m_array,log10Lambda_m_array=get_kde_2D([log10_Lambda,m],bin_num)
Lambda_beta6_array,m_array,Lambda_beta6_m_array=get_kde_2D([Lambda_beta6,m],bin_num)
radius_array,m_array,radius_m_array=get_kde_2D([radius,m],bin_num)
log10_Lambda1_array,log10_Lambda2_array,log10_Lambda1_log10_Lambda2_array=get_kde_2D([log10_Lambda1,log10_Lambda2],bin_num)
q_array,log10_Lambda_binary_array,q_log10_Lambda_binary_array=get_kde_2D([q,log10_Lambda_binary],bin_num)
q_array,Lambda_binary_beta_m_chirp_radius_onepointfour6_1000_array,q_Lambda_binary_beta_m_chirp_radius_onepointfour6_1000_array=get_kde_2D([q,Lambda_binary_beta_m_chirp_radius_onepointfour6_1000],bin_num)
m_chirp_array,log10_Lambda_binary_array,m_chirp_log10_Lambda_binary_array=get_kde_2D([m_chirp,log10_Lambda_binary],bin_num)

import scipy.optimize as opt
def plot_density_1D(x,density_array,percentile,color_list,ax,marginal_axis='x',unit='',legend_loc=0):
    n = 20
    density_array_max=density_array.max()
    t = np.linspace(0, density_array_max, n)
    integral = ((density_array >= t[:, None]) * density_array).sum(axis=(1))
    f = interpolate.interp1d(integral, t)
    t_contours = f(np.array(percentile))
# =============================================================================
#     f = interpolate.interp1d(x,density_array)
#     opt.newton(f,t_contours,args=[])
#     print f(t_contours)
# =============================================================================
    x_contours = []
    density_countours = []
    for index_list_flag in density_array>t_contours[:,None]:
        index_list=np.where(index_list_flag)[0]
        x_contours.append(x[[index_list.min(),index_list.max()]])
        density_countours.append(density_array[[index_list.min(),index_list.max()]])
    if(marginal_axis=='x'):
        ax.plot(x,density_array,linewidth=5*figsize_norm)
        for i in range(len(percentile)):
            ax.plot([x_contours[i][0],x_contours[i][0]],[0,density_countours[i][0]],'--',color=color_list[i],linewidth=5*figsize_norm)
            ax.plot([x_contours[i][1],x_contours[i][1]],[0,density_countours[i][1]],'--',color=color_list[i],linewidth=5*figsize_norm,label='%.2f - %.2f'%(x_contours[i][0],x_contours[i][1])+unit)
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.legend(fontsize=20*figsize_norm,frameon=False,loc=legend_loc)
    elif(marginal_axis=='y'):
        ax.plot(density_array,x,linewidth=5*figsize_norm)
        for i in range(len(percentile)):
            ax.plot([0,density_countours[i][0]],[x_contours[i][0],x_contours[i][0]],'--',color=color_list[i],linewidth=5*figsize_norm)
            ax.plot([0,density_countours[i][1]],[x_contours[i][1],x_contours[i][1]],'--',color=color_list[i],linewidth=5*figsize_norm,label='%.2f - %.2f'%(x_contours[i][0],x_contours[i][1])+unit)
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.legend(fontsize=20*figsize_norm,frameon=False,loc=legend_loc)

import matplotlib.gridspec as gridspec
def plot_density_2D(xx,yy,density_array,percentile,color_list,label_x,label_y,x_unit='',y_unit='',legend_loc=[0,0,0]):
    density_array = density_array / (density_array.sum())#*4*half_bin_size_x*half_bin_size_y)
    f = plt.figure(figsize=(20*figsize_norm,20*figsize_norm))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4],wspace=0.05,hspace=0.05)
    xy_density = plt.subplot(gs[2])
    xy_density.tick_params(labelsize=40*figsize_norm)
    n = 20
    t = np.linspace(0, density_array.max(), n)
    integral = ((density_array >= t[:, None, None]) * density_array).sum(axis=(1,2))
    f = interpolate.interp1d(integral, t)
    t_contours = f(np.array(percentile))
    label_contours = np.array(100*np.array(percentile)).astype('str')
    print label_contours
    cs = xy_density.contour(xx,yy,density_array, t_contours,linestyles='--',colors=color_list,linewidths=5*figsize_norm)
    cf = xy_density.contourf(xx, yy, density_array,100, cmap='Blues')
    #cf.colorbar()
    peak_density=np.where(density_array==density_array.max())
    xy_density.plot([xx[peak_density]],[yy[peak_density]],'+')
    fmt = {}
    print cs.levels
    for i in range(len(cs.levels)):
        fmt[cs.levels[i]] = label_contours[i]+'%'
        cs.collections[i].set_label(label_contours[i]+'%')
    xy_density.clabel(cs, cs.levels, inline=True, fmt=fmt, fontsize=30*figsize_norm)
    xy_density.set_xlabel(label_x,fontsize=40*figsize_norm)
    xy_density.set_ylabel(label_y,fontsize=40*figsize_norm)
    xy_density.legend(fontsize=40*figsize_norm,frameon=False,loc=legend_loc[0])
    x_marginal = plt.subplot(gs[0],sharex=xy_density)
    plot_density_1D(xx[:,0],density_array.sum(1),percentile,color_list,x_marginal,marginal_axis='x',unit=x_unit,legend_loc=legend_loc[1])
    
    y_marginal = plt.subplot(gs[3],sharey=xy_density)
    plot_density_1D(yy[0],density_array.sum(0),percentile,color_list,y_marginal,marginal_axis='y',unit=y_unit,legend_loc=legend_loc[2])
    
    xy_density.set_xlim(xx[0,0],xx[-1,0])
    xy_density.set_ylim(yy[0,0],yy[0,-1])
    return xy_density

percentile_list = [0.997, 0.95, 0.68, 0.4, 0.2]
color_list = ['c','r','k','y','b','g']
plot_density_2D(log10_Lambda_array,m_array,log10Lambda_m_array,percentile_list,color_list,'log$_{10}\Lambda$','mass(M$_\odot$)',y_unit='M$_\odot$')
plot_density_2D(radius_array,m_array,radius_m_array,percentile_list,color_list,'radius(km)','mass(M$_\odot$)')
plot_density_2D(log10_Lambda1_array,log10_Lambda2_array,log10_Lambda1_log10_Lambda2_array,percentile_list,color_list,'log$_{10}\Lambda_1$','log$_{10}\Lambda_2$',legend_loc=[4,2,1])
plot_density_2D(q_array,log10_Lambda_binary_array,q_log10_Lambda_binary_array,percentile_list,color_list,'q','log$_{10}\\tilde\Lambda$',legend_loc=[2,2,1])
plot_density_2D(q_array,Lambda_binary_beta_m_chirp_radius_onepointfour6_1000_array,q_Lambda_binary_beta_m_chirp_radius_onepointfour6_1000_array,percentile_list,color_list,'q','$1000\\tilde\Lambda (M_{ch}G/R_{1.4}c^2)^6$',legend_loc=[1,2,1])
plot_density_2D(m_chirp_array,log10_Lambda_binary_array,m_chirp_log10_Lambda_binary_array,percentile_list,color_list,'$\mathcal{M}$(M$_\odot$)','log$_{10}\\tilde\Lambda$',x_unit='M$_\odot$')


def Low_tidal_cutoff_MC(mass,maxmass):
    mass_over_maxmass=mass/maxmass
    return np.exp(13.42-23.04*mass_over_maxmass+20.56*mass_over_maxmass**2-9.615*mass_over_maxmass**3)
def Low_tidal_cutoff_UG(mass):
    mass_over_maxmass=mass
    return np.exp(18.819-19.862*mass_over_maxmass+10.881*mass_over_maxmass**2-2.5713*mass_over_maxmass**3)

mass_cutoff=np.linspace(1.0,1.8,100)
density_plot=plot_density_2D(log10_Lambda,m,log10Lambda_m_array,bin_num,[0.95,0.9, 0.75, 0.5, 0.25, 0.1],'log$_{10}\Lambda$','mass')
density_plot.plot(np.log10(Low_tidal_cutoff_MC(mass_cutoff,2.0)),mass_cutoff,label='maximum compact lower bound')
density_plot.plot(np.log10(Low_tidal_cutoff_MC(mass_cutoff,4.0)),mass_cutoff,label='maximum compact upper bound')
density_plot.plot(np.log10(Low_tidal_cutoff_UG(mass_cutoff)),mass_cutoff,label='unitary gas lower bound')
density_plot.legend(fontsize=10)



!pycbc_inference_plot_posterior\
    --input-file uniform_mass_prior_common_eos_posteriors.hdf dns_mass_prior_common_eos_posteriors.hdf \
    galactic_ns_mass_prior_common_eos_posteriors.hdf \
    --output-file radius_lambda_tilde_posterior.png \
    --plot-marginal \
    --plot-contour \
    --contour-percentiles 90 \
    --parameters \
            'lambda_tilde(mass1, mass2, lambdasym*((mass2/mass1)**3), lambdasym*((mass1/mass2)**3)):$\tilde{\Lambda}$' \
            '11.2*(mchirp/(1+redshift(40.7)))*((lambda_tilde(mass1, mass2, lambdasym*((mass2/mass1)**3), lambdasym*((mass1/mass2)**3)))/800)**(1./6.):$\hat R$' \
    --mins '11.2*(mchirp/(1+redshift(40.7)))*((lambda_tilde(mass1, mass2, lambdasym*((mass2/mass1)**3), lambdasym*((mass1/mass2)**3)))/800)**(1./6.):7'\
    --maxs 'lambda_tilde(mass1, mass2, lambdasym*((mass2/mass1)**3), lambdasym*((mass1/mass2)**3)):1500' \
        '11.2*(mchirp/(1+redshift(40.7)))*((lambda_tilde(mass1, mass2, lambdasym*((mass2/mass1)**3), lambdasym*((mass1/mass2)**3)))/800)**(1./6.):15.0'\
    --input-file-labels "Uniform distribution" "Double Neutron Stars" "Galactic Neutron Stars" \
    --verbose
