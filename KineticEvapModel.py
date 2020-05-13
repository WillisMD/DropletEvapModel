#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple kinetic model of isothermal continuous evaporation from a sphere
Assumes ideal solutions (activity coefficients of unity)

@author: meganwillis
(Adapted from pyvap by @awbirdsall)
"""

################################
from __future__ import division
from math import e
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import pi, k, R, N_A
from scipy.integrate import ode
################################

#########################
###---INPUTS--:##########
#########################

maleic = {"name": "Maleic",
        "Dg": 7.2e-6, #m^2/s
        "M": 0.1161, #kg/mol
        "rho": 1590.0, #kg/m3
        "cinf": 0,
        "p298": 1.7e-3, #Pa
    "delh": 115e+3} #J/mol

#select which compounds to use in the model
cmpds = [maleic] #comma separated list

comp = [1] # mole fraction of non-water species in the droplet (calculate using AIOMFAC or E-AIM)
r_init = 20e-6 # starting radius, m
time = 500 # integration time, s
numpts = 5000 # number of points to integrate
temp = 298.15 # K
has_water=False #True for RH > 0%
xh2o=None #None or float, constant for a stable RH, calculate using AIOMFAC
normplot = False #normalize output plot of total molecules to 1?
plotinseconds = True #False plots in hours
calcevaprate = True

#########################
###---CONSTANTS--:#######
#########################

MA_H2O = 0.018/6.02e23 # mass water, kg molec^-1
RHO_H2O = 1.000e3 # density water, kg m^-3

#########################
###---FUNCTIONs--:#######
#########################

def dn(t, y, Ma, rho, cinf, po, Dg, T, has_water=False, xh2o=None):
    '''Construct differential equations to describe change in n
    (1) Input parameters:
    t (float)
        Time
    y (np.array)
        1D-array of number of molecules of each component
    Ma (np.array)
        1D-array of molecular weights, kg/molec
    rho (np.array)
        1D-array of densities, kg/m3 
    cinf (np.array)
        1D-array of concentration of substance at infinite distance, molec/m3 
    po (np.array)
        1D-array of pure compound vapor pressures, Pa
    Dg (np.array)
        1D-array of gas-phase diffusivities, m^2/s 
    T (float)
        Temperature, K
    has_water (Boolean)
        Include water in calculation.
    xh2o (float)
        Fixed water mole fraction. Only used if has_water is True.
    ------------------
    (2) Returns
    dn (np.array)
        1D-array of dn for all components'''
    ytot = y.sum()
    v = y*Ma/rho # array of partial volumes, m^3    
    if has_water:
        # known mole fraction of water
        ntot = 1/(1-xh2o) * ytot
        nh2o = xh2o * ntot
        vh2o = nh2o * MA_H2O / RHO_H2O
        vtot = v.sum()+vh2o # particle volume, m^3
    else:
        ntot = ytot
        vtot = v.sum()

    r = (3*vtot/(4*pi))**(1/3) # radius, m^3

    x = y/ntot # mole fractions, where ntot includes h2o if present
    # assume ideality in vapor pressure calculation
    cs = x*po/(k*T) # gas-phase concentration at surface, molec m^-3
    # array of differential changes in number of molecules
    delta_n = 4*pi*r*Dg*(cinf-cs)
    return delta_n

def evaporation_model(components, ninit, T, num, dt, has_water=False, xh2o=None):
    '''Calculate evaporation of multicomponent particle
    (1) Input parameters:
    components (list)
        List of components of particles, each represented as a dict of parameters
    ninit (np.ndarray)
        1D array of starting number of molecules of each component in components
    T (float)
        Temperature of evaporation in K
    num (int)
        Total number of integrated time points, including t=0.
    dt (float)
        Integration timestep in seconds
    has_water (boolean (optional, default False))
        Whether presence of water is considered in calculating evaporation (separate from list of components)
    xh2o (float (optional, default None))
        Fixed mole fraction of water to include in particle. Only considered if has_water is True.
    ------------------
    (2) Returns
    output (np.ndarray)
        2D array of results from integration: number of molecules of each component
        remaining at each timestep. First index is along num timesteps and second
        index is along len(components)'''
    # extract data from components and create empty data arrays
    Ma = np.empty(len(components))
    rho = np.empty(len(components))
    cinf = np.empty(len(components))
    po = np.empty(len(components))
    Dg = np.empty(len(components))
    for i, component in enumerate(components):
        # use Ma (molecular weight, kg/molec) if provided, otherwise use M (molar weight, kg/mol)
        if 'Ma' in component:
            Ma[i] = component['Ma']
        else:
            Ma[i] = component['M']/N_A
        rho[i] = component['rho']
        cinf[i] = component['cinf']
        # use p298 and delh if available. otherwise use p0_a and p0_b
        if ('p298' in component) and ('delh' in component):
            p0a, p0b = convert_p0_enth_a_b(component['p298'],
                                           component['delh'], 298.15)
            po[i] = calc_p(p0a, p0b, T)
        else:
            po[i] = calc_p(component['p0_a'], component['p0_b'], T)
        Dg[i] = component['Dg']

    # set up ode
    output = np.empty((int(num), len(components)))
    output[0, :] = ninit

    r = ode(dn).set_integrator('lsoda', with_jacobian=False,)
    r.set_initial_value(ninit, t=0)
    r.set_f_params(Ma, rho, cinf, po, Dg, T, has_water, xh2o)

    # integrate and save output
    entry = 0
    # use `entry` condition to avoid rounding errors causing possible problems with `r.t` condition
    while r.successful() and entry < num-1:
        entry = int(round(r.t/dt))+1
        nextstep = r.integrate(r.t + dt)
        output[entry, :] = nextstep

    return output


def calc_radius(components, ns, has_water=False, xh2o=None):
    '''Given array of n values in time and list of components, calculate radii.
    (1) Input parameters:
    components (list)
        List of dicts for each component
    ns (np.array)
        2D array of molar amounts of material. First index is timestep, second index is index of component in components
    has_water (Boolean (optional, default False))
        Whether water is added in addition to components
    xh2o (float (optional, default None))
        Fixed mole fraction of water added to particle in calculating value. Only considered if has_water is True.
    ------------------
    (2) Returns:
    r (np.array)
        Array of radii, in meters, for each row of components given in `ns`.
    '''
    vtot = np.zeros_like(ns.shape[0])
    for i, c in enumerate(components):
        # convert number of molecules in ns, using molecular or molar mass (Ma or M)
        # (units kg (molec or mol)^-1) and density rho (kg m^-3), to volume in m^3
        if 'Ma' in c:
            v = ns[:, i] * c['Ma'] / c['rho']
        else:
            v = ns[:, i] * c['M'] / c['rho'] / N_A
        vtot = vtot + v
    if has_water:
        # fixed xh2o for fixed RH (water activity)
        nh2o = xh2o/(1-xh2o)*ns.sum(axis=1)
        vh2o = nh2o * MA_H2O / RHO_H2O
        vtot = vtot + vh2o
    r = (3*vtot/(4*pi))**(1/3)
    return r


def convert_p0_enth_a_b(p0, delta_H, t0):
    '''Converts p0 and deltaH to vapor pressure temp dependence parameters.
    (1) Input parameters:
    p0 (float or ndarray)
        Vapor pressure at reference temperature in Pa.
    delta_H (float or ndarray)
        Enthalpy of vaporization (or sublimation) in J/mol.
    t0 (float or ndarray)
        Reference temperature for p0 value in K.
    ------------------
    (2) Returns:
    p0_a, p0_b (float)
        a (intercept, Pa) and b (slope, 1000/K) linear regression parameters for
        temperature dependence of log10(vapor pressure).'''
    p0_a = 1/np.log(10) * ((delta_H/(R*t0)) + np.log(p0))
    p0_b = -delta_H/(1000*np.log(10)*R)
    return p0_a, p0_b


def calc_p(a, b, temp):
    '''Given regression line parameters, calculate vapor pressure at given temperature.'''
    log_p = a + b*(1000./temp)
    p = pow(10, log_p)
    return p


def plot_evap(x, molec_data, r_data, series_labels, xlabel, normplot):
    fig, ax = plt.subplots()
    if series_labels is not None:
        for i in np.arange(molec_data.shape[1]):
            if normplot == True:
                ax.plot(x, molec_data[:, i]/molec_data[0,i], label=series_labels[i]) #normalize to molecules
            else:
                ax.plot(x, molec_data[:, i], label=series_labels[i]) #absolute molecules
        ax.legend(loc='lower right', ncol=3)
    else:
        ax.plot(x, molec_data[:, i])
    if normplot is True:
        ax.set_ylabel("normalized molecules remaining")
    else:
        ax.set_ylabel("number of molecules remaining")
    ax.set_xlabel(xlabel)
    if normplot is True:
        ax.set_ylim([0,1.1])
    else:
        ax.set_ylim(0, np.max(molec_data[0, :])*1.1)

    ax2 = ax.twinx()
    ax2.plot(x[:-1], r_data[:-1]*(1e6), 'k--', label='radius (right axis)') #plot radius, converting to um
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax2.set_ylabel("particle radius (microns)")
    ax2.set_ylim(0, r_data[0]*(1e6)*1.1)
    ax2.legend()

    return fig, (ax, ax2)


def efold_time(t, y):
    '''Calculate an e-folding time given timeseries and corresponding y values. Return -1 if e-fold time not reached in given series.
    Assume that e-folding time is sensical values for data series (i.e., monotonic decay) and that value at t=inf is 0.'''
    efold = np.where(y <= 1./e*y[0])[0]
    if efold.size > 0:
        efold_time = t[efold[0]]
    else:
        efold_time = -1
    return efold_time


#########################
####-----MAIN----:#######
#########################

output_dict = dict()
# calc initial total number of molecules
avg_rho = np.average([x['rho'] for x in cmpds], weights=comp) # kg/m3 
total_mass = 4./3.*pi * r_init**3 * avg_rho # kg
# use either Ma or M
get_Ma = lambda cmpd: cmpd['Ma'] if 'Ma' in cmpd else cmpd['M']/N_A
avg_molec_mass = np.average([get_Ma(x) for x in cmpds],weights=comp) # kg/molec
total_molec = total_mass/avg_molec_mass # molec

if type(comp)==list:
    comp = np.array(comp)
ncomp = comp/comp.sum() # make sure composition is normalized to 1
molec_init = ncomp*total_molec

# set up and integrate ODE
t_series, dt_evap = np.linspace(start=0, stop=time, num=numpts, retstep=True) 
#t_series is an array of timesteps corresponding to evap_comp output
evap_comp = evaporation_model(cmpds, ninit=molec_init, T=temp, num=numpts, dt=dt_evap, has_water=has_water, xh2o=xh2o)
output_dict.update({'t_series': t_series, 'evap_comp': evap_comp})

# calculte radius timeseries
radius = calc_radius(cmpds, evap_comp, has_water, xh2o)
output_dict.update({'radius': radius})

# plot
complabels = [x['name'] for x in cmpds]
if plotinseconds == True:
    xlabel = "time (seconds)"
    fig, (ax, ax2) = plot_evap(x=output_dict['t_series'], #plot in seconds
                               molec_data=evap_comp,
                               r_data=radius,
                               series_labels=complabels,
                               xlabel=xlabel,
                               normplot=normplot)
else:
    xlabel = "time (hours)"
    fig, (ax, ax2) = plot_evap(x=output_dict['t_series']/3600, #convert seconds to hours
                               molec_data=evap_comp,
                               r_data=radius,
                               series_labels=complabels,
                               xlabel=xlabel,
                               normplot=normplot)
output_dict.update({'evap_fig': (fig, (ax, ax2))})

# e-folding times, converted from seconds to hours
efold_dict = {l: efold_time(output_dict['t_series']/3600, evap_comp[:, i])
              for i, l in enumerate(complabels)}
output_dict.update({'efold_dict': efold_dict})

# display figure
plt.show()

# save figure
evap_fig, (evap_ax0, evap_ax1) = output_dict["evap_fig"]
evap_fig.savefig("evaporation.png")

# back out evaporation rates
if calcevaprate == True:
    time = output_dict['t_series']
    n_part = output_dict['evap_comp']
    n_evap = n_part[0,:]-n_part
    
    rate = np.empty(n_part.shape)
    for i in range(len(n_part)-1):
            rate[i,:] = (n_evap[i+1,:]-n_evap[i,:])/(time[i+1]-time[i])
    np.savetxt("evaproation_rate.csv", rate, delimiter=",")      
#    plt.plot(time, rate)
#    plt.ylim([0,10e13])
#    plt.xlim([-1,30])
    
    
# save csv of evaporation model output (no. of molecules of each component, at each timestep)
np.savetxt("evaporation_molecs.csv", output_dict["evap_comp"], delimiter=",")
np.savetxt("evaporation_radius.csv", output_dict["radius"], delimiter=",")
np.savetxt("evaporation_time.csv", output_dict["t_series"], delimiter=",")
