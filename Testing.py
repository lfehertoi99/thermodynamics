# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:20:42 2020

@author: lilif
"""
#%%
import numpy as np
from Ball import Ball
import pylab as pl
from itertools import combinations
import matplotlib.pyplot as plt
import unittest
#%%
'''
Test cases for classes/methods used in simulation
'''
class TestSimulation(unittest.TestCase):
    #test repr and str for Ball object
    def test_repr(self):
        ball = Ball(np.array([2, 3]), np.array([1, 2]))
        self.assertEqual(ball.__repr__(), 
            "Ball(pos = [2 3], vel = [1 2], rad = 1, mass = 1)")
    
    #test move method
    def test_move(self):
        ball = Ball(np.array([4, 5]), np.array([5, 6]))
        ball.move(5)
        self.assertEqual(ball.getpos().all(), np.array([29, 35]).all(), \
            "should be [29, 35]")
        self.assertEqual(ball.getvel().all(), np.array([5, 6]).all(),\
                         "should be [5, 6]")
    
    #test for case of ball colliding with container with ball velocity in x
    #dirn only
    def test_cont_coll(self):
        ball = Ball(pos = np.array([-5, 0]), vel = np.array([1, 0]))
        cont = Ball(pos = np.array([0, 0]), vel = np.array([0, 0]), \
                    rad = -10, mass = 1e10, b_type = "container")
        self.assertEqual(ball.time_to_collision(cont), 14, "should be 14")
        
        ball.move(ball.time_to_collision(cont))
        c_vel = ball.collide(cont)[0]
        print(c_vel)
        ball.setvel(c_vel[0])
        #cont.setvel(c_vel[1]); container velocity should stay zero
        self.assertEqual(ball.getpos().all(), np.array([9, 0]).all(), \
                         "should be [9, 0]")
        self.assertEqual(ball.getvel().all(), np.array([-1, 0]).all(), \
                         "should be [-1, 0]")
        self.assertEqual(cont.getpos().all(),  np.array([0, 0]).all(),\
                         "should be [0, 0]")
        self.assertEqual(cont.getvel().all(), np.array([0, 0]).all(),\
                       "should be [0, 0]")
            
    #test for two balls traveling towards each other along x-axis
    def test_x_coll(self):
        ball1 = Ball(np.array([-2, 0]), np.array([1, 0]), rad = 1, mass = 1)
        ball2 = Ball(np.array([2, 0]), np.array([-1, 0]), rad = 1, mass = 1)
        self.assertEqual(ball1.time_to_collision(ball2), 1.0, "should be 1.0") 
        
    #test for two balls traveling toward each other diagonally
    def test_diag(self):
        ball1 = Ball(np.array([0, 0]), np.array([1, -1]), rad = 1, mass = 1)
        ball2 = Ball(np.array([4, 0]), np.array([-1, -1]), rad = 1, mass = 1)
        self.assertEqual(ball1.time_to_collision(ball2), 1.0, "should be 1.0")
        
        c_vel = ball1.collide(ball2)[0]
        ball1.setvel(c_vel[0])
        ball2.setvel(c_vel[1])
        
        self.assertEqual(ball1.getvel().all(), np.array([-1, -1]).all(), \
            "should be [-1, -1]")
        self.assertEqual(ball2.getvel().all(), np.array([1, -1]).all(), \
            "should be [1, -1]")
            
    #test method for kinetic energy
    def test_ke(self):
        ball = Ball(np.array([0, 0]), np.array([1, 3]), rad = 1, mass = 2)
        self.assertEqual(ball.kin_e(), 10.0, "should be 10.0")
        
    #test conservation of ke in collision in many-body system
    def test_coe(self):
        sim = Simulation(100, 0.1)
        a = sim.sys_kin_e()
        sim.next_collision1() #evolve 1 step forward
        b = sim.sys_kin_e()
        self.assertEqual(np.around(a, 6), np.around(b, 6), \
                         "should be equal before and after collision")
    
    def mfp(self):
        sim = Simulation(10, density = 0.1)
        sim.run1(15)
        self.assertIsInstance(np.mean(sim._paths), float)

if __name__ == "__main__":
    unittest.main()

    


