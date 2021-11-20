# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:12:12 2020

@author: lilif
"""
#%%
import numpy as np 
from numpy import corrcoef
from itertools import combinations
from Ball import Ball
from Simulation import Simulation
import scipy.optimize as spo
from scipy.stats import norm
#%%
def pressure_temperature(iterations, points, densities):
    '''
    Returns data from simulations to look at the relationship between 
    temperature and pressure. Can also generate plots of p against T.

    Parameters
    ----------
    iterations : int
        number of times to run entire simulation. more iterations results in a
        lower statistical uncertainty of the results.
    points : int
        number of points in temperature range to test.
    densities: list
        list of denisities to run method for

    Returns
    -------
    temperatures : array
        array containing temperatures which have been used in simulation.
    pressures : array
        array containing pressures which have been measured in the simulations.

    '''
    fig, ax = plt.subplots()
    ax.set_title("Equilibrium pressure vs. Temperature")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("Pressure")
    
    for d in densities:
        orig_temperatures = np.linspace(1, 8, points)
        temperatures = np.zeros(points)
        pressures = np.zeros(points)
        
        for i in range(len(orig_temperatures)):
            for k in range(iterations):
                test_sim = Simulation(50, density = d, \
                                      temp = orig_temperatures[i])
                p, t = test_sim.run2(250, plots= False)
                pressures[i] += p
                temperatures[i] = t #change to actual measured temperature
                
        pressures /= iterations
        
        #fit line to data
        X = temperatures
        Y = pressures
        coef, cov = np.polyfit(X, Y, 1, cov = True)
        print("gradient: %.3f, variance of gradient: %.3f" % \
              (coef[0], cov[0][0]))
        poly1d_func = np.poly1d(coef)

        #plot
        ax.scatter(temperatures, pressures)
        ax.plot(X, poly1d_func(X), label = ("density:%.3f" % d))
    ax.legend()

    return temperatures, pressures
#%%
def pressure_density(iterations, points):
    '''
    Returns data from simulations to look at relationship between density and
    pressure. Can also generate plots of p against rho. 

    Parameters
    ----------
    iterations : int
        number of times to run entire simulation. more iterations results in a
        lower statistical uncertainty of the reusults.
    points : int
        number of points in temperature range to test.

    Returns
    -------
    densities : array
        array containing temperatures which have been used in simulation.
    pressures : array
        array containing pressures which have been measured in the simulation.

    '''
    
    densities = np.linspace(0.01, 0.1, points)
    pressures = np.zeros(points)
    
    for i in range(len(densities)):
        for k in range(iterations):
            test_sim = Simulation(50, density = densities[i])
            p, t = test_sim.run2(200, plots = False)
            pressures[i] += p
    
    pressures /= iterations
    
    X = densities
    Y = pressures
    coef = np.polyfit(X, Y, 1)
    poly1d_func = np.poly1d(coef)
    
    #plot
    fig, ax = plt.subplots()
    ax.set_title("Equilibrium pressure vs. Density, Temperature = 1")
    ax.set_xlabel("Density")
    ax.set_ylabel("Pressure")
    ax.scatter(densities, pressures)
    ax.plot(X, poly1d_func(X))
    
    return densities, pressures
#%%
def velocities_temperature(points):
    '''
    Obtain velocity distribution at different temperatures. Fit 2d Maxwell
    distribution and obtain fit parameters. Print fit parameters and compare
    to theoretically calculated values.
    
    Parameters
    ----------
    points : int
        number of temperature values in range to run simulation for

    Returns
    -------
    array
        contains parameters of each maxwell distribution fitted to data

    '''
    temperatures = np.linspace(2, 5, points)
    
    fig, ax = plt.subplots()
    ax.set_title("Normalized velocity distribution for different temperatures")
    ax.set_xlabel("Velocity")
    ax.set_ylabel("Probability")
    
    params = [] 
    
    for i in range(len(temperatures)):
        
        test_sim = Simulation(100, temp = temperatures[i])
        test_sim.run2(350, plots = False)
        
        
        hist, bins = np.histogram(test_sim._velocities, bins = 10,\
                                  density = True)
        X = (bins[:-1] + bins[1:])/2
        Y = hist
        
        def maxwell(x, a, b):
            return a * x * norm.pdf(x, scale = b)
        
        popt, pcov = spo.curve_fit(maxwell, X, Y)
        params += [*popt]
        actual_temp = test_sim.sys_kin_e()/test_sim._n_balls
        
        print("temperature:%.3f" % 
              actual_temp, "calculated:%.3f"
              % (np.sqrt(actual_temp)), "actual: %.3f" %
              params[-1],"uncertainty: %.3f" %
              ((np.std(test_sim._velocities))/np.sqrt(350)))
            
        X_new = np.linspace(bins[0], bins[-1], 100)
        
        ax.plot(X_new, maxwell(X_new, *popt))
        ax.hist(test_sim._velocities, bins = 10, density = True, \
                alpha = 0.5, label = ("T = %.2f" % actual_temp))
        
    ax.legend()
    return params      
#%%
def velocities_mass(points):
    '''
    Obtain velocity distribution at different ball masses. Fir 2d Maxwell
    distribtuion and obtain fit parameters. Print fit parameters and compare
    to theoretically calculated values.
    

    Parameters
    ----------
    points : int
     number of mass values in range to run simulation for

    Returns
    -------
    array
        contains parameters of each macwell distribution fitted to data

    '''
    masses = np.linspace(0.5, 10, points)
    
    fig, ax = plt.subplots()
    ax.set_title("Normalized velocity distribution for different masses")
    ax.set_xlabel("Velocity")
    ax.set_ylabel("Probability")
    
    params = [] 
    
    for i in range(len(masses)):
        
        test_sim = Simulation(50, temp = 1/masses[i])
        test_sim.run(350, plots = False)
        
        
        hist, bins = np.histogram(test_sim._velocities, bins = 10, \
                                  density = True)
        X = (bins[:-1] + bins[1:])/2
        Y = hist
        
        def maxwell(x, a, b):
            return a * x * norm.pdf(x, scale = b)
        
        popt, pcov = spo.curve_fit(maxwell, X, Y)
        params += [*popt]
        X_new = np.linspace(bins[0], bins[-1], 100)
        
        ax.plot(X_new, maxwell(X_new, *popt), label = ("m = %.2f" % masses[i]))
        ax.hist(test_sim._velocities, bins = 10, density = True, alpha = 0.5)
        
    ax.legend()
    
    return params      
#%%
def vd_waals(n_balls, density, ball_radii, iterations, points):
    '''
    Obtain parameter of Van der Waals fit by looking at slope of the pot of 
    pressure against temperature. The vdW parameter a is taken to be zero 
    since there are no attractive forces between the hard spheres.

    Parameters
    ----------
    n_balls: int
        number of balls in simulation
    density : float
        density of balls to use in simulation
    ball_radii : list
        radii of balls to use in simulation
    iterations : int
        iterations : int
        number of times to run entire simulation. more iterations results in a
        lower statistical uncertainty of the results.
    points : int
        number of temperature values in range to run simulation for
    

    Returns
    -------
    array
        contains temperatures for which data was taken
    array
        contains pressured mesured at each temperature

    '''
    params = []
    for r in ball_radii:
        #do simulation for different temperatures
        #get pressure temperature relationship for each case
        number = n_balls
        volume = (n_balls * (r**2) * np.pi) / density
        
        
        temperatures = np.linspace(1, 10, points)
        pressures = np.zeros(points)
        
        for k in range(iterations):
            for i in range(len(temperatures)):    
                test_sim = Simulation(50, density = density, \
                                      temp = temperatures[i])
                p, t = test_sim.run2(150, plots= False)

                pressures[i] += p
                temperatures[i] += t
                
        pressures /= iterations

        #fit line to data
        X = temperatures
        Y = pressures
        
        def func(x, b):
            nonlocal number
            nonlocal volume
            return (x * (volume - (number*b)))
        
        popt, pcov = spo.curve_fit(func, X, Y)
        params.append(popt[0])
        
    fig, ax = plt.subplots()
    ax.set_title("Van der Waals parameter b as function of ball area")
    ax.set_xlabel("Ball area")
    ax.set_ylabel("Optimal value of b")
    ax.scatter(ball_radii, params)
        
    return [r**2 for r in ball_radii], params
#%%
def mean_free_path(iterations, points):
    
    densities = np.linspace(0.02, 0.25, points)
    mfp = np.zeros(points)
    
    for k in range(iterations):
        for i in range(len(densities)):
            sim = Simulation(50, density = densities[i])
            sim.run2(200)
            mfp[i] += np.mean(sim._paths)
            
    mfp /= iterations
    
    X = densities
    Y = mfp
    
    def func(x, a):
        return a/x
    
    popt, pcov = spo.curve_fit(func, X, Y)
    
    X_ax = np.linspace(0.02, 0.25, 200)
    fig, ax = plt.subplots()
    ax.set_title("Desnity dependence of mean free path")
    ax.set_xlabel("Density")
    ax.set_ylabel("Mean free path")
    ax.scatter(densities, mfp)
    ax.plot(X_ax, func(X_ax, *popt))
    
    return densities, mfp































    