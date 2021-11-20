# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:12:12 2020

@author: lilif
"""
#%%
import numpy as np 
from itertools import combinations
from Ball import Ball
from Simulation import Simulation
import scipy.optimize as spo
from scipy.stats import norm
#%%
sim = Simulation(50, 0.1)
sim.run2(200, plots = True)
#%%
a, b = vd_waals(50, 0.15, [0.5, 0.6, 0.7, 0.8], 3, 5)
print(a, b)
#%%
pressure_temperature(4, 5, [0.05, 0.1, 0.15])
#%%
velocities_temperature(4)
#%%
d, p = mean_free_path(3, 5)
#%%
sim = Simulation(50, 0.1)
print(sim.sys_mom())
























    