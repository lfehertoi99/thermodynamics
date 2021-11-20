# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:47:24 2020

@author: lilif
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
#%%
class Ball:
    ''' 
    Ball class to represent particles taking part in collision.
    -----------------------------------------------------------
    Attributes:
        - pos: position, numpy array of shape (1, 2)
        - vel: velocity, numpy array of shape (1, 2)
        - b_type: ball type (ball or container), str
        - rad: radius, float
        - mass: float
        -clock: time since previous collision
    -----------------------------------------------------------    
    Methods:
        - getpos, getvel: access methods for position and velocity
        - setpos, setvel: modifier methods for position and velocity
        - move(t): move ball forward in time by t
        - time_to_collision(other): calculate time to collision with other ball
        - collide(other): calculate post-collision velocities for both balls
        - get_patch: get patch to create animation
    -----------------------------------------------------------    
    '''
    def __init__(self, pos, vel, b_type = "ball", rad = 1.0, mass = 1.0):
        '''
        Parameters
        ----------
        pos : array
            position of ball
        vel : array
            velocity of ball
        b_type : str, optional
            type of ball (ball or container). The default is "ball".
        rad : float, optional
            radius of ball The default is 1.0.
        mass : float, optional
            mass of ball. The default is 1.0.
            
        Returns
        -------
        None.
        '''
        self._pos = pos
        self._vel = vel
        self._b_type = b_type
        self._rad = rad
        self._mass = mass
        
        self._clock = 0
        
    def __repr__(self):
        '''
        Returns
        -------
        string
            string representation which can be used to recreate object.
        '''
        return "Ball(pos = %s, vel = %s, rad = %g, mass = %g)" \
                % (self._pos, self._vel, self._rad, self._mass)
        
    def getpos(self):
        '''
        Access method for ball position.
        Returns
        -------
        array
            position of ball
        '''
        return self._pos
    
    def setpos(self, pos):
        '''
        Modifier method for ball position
        Parameters
        ----------
        pos : array, shape = (2, 1)
            desired new position of ball
        Returns
        -------
        None.
        '''
        self._pos = pos
    
    def getvel(self):
        '''
        Access method for ball velocity
        Returns
        -------
        array
            velocity of ball
        '''
        return self._vel
    
    def setvel(self, vel):
        '''
        Modifier method for ball velocity
        Parameters
        ----------
        vel : array, shape = (2, 1)
            desired new velocity of ball

        Returns
        -------
        None.
        '''
        self._vel = vel
        
    def kin_e(self):
        return 0.5 * (np.dot(self._vel, self._vel) * self._mass)
    
    def move(self, dt):
        '''
        Move ball forward in time by t.
        Parameters
        ----------
        dt : float
            amount of time to move ball forward by.

        Returns
        -------
        None.
        '''
        newpos = self._pos + self._vel * dt
        self.setpos(newpos)
        
    def time_to_collision(self, other):
        '''
        Calculates time to collision with other ball. This is done by solving \
        a quadratic equation obtained from considering the positions of the \
        balls after time t. The discriminant is calculated first to discard \
        complex solutions, and negative (past) solutions are also discarded.

        Parameters
        ----------
        other : Ball object
            The other ball taking part in the collision

        Returns
        -------
        t : float
            time until the two balls collide with each other.
        '''
        #truncate floats to avoid issues
        r = np.round(self._pos - other._pos, 6) #relative position
        v = np.round(self._vel - other._vel, 6) #relative velocity
        coll_rad = np.round(self._rad + other._rad, 6) #collision distance
        
        #coefficients of quadratic equation for dt
        a, b, c = np.dot(v, v), 2*np.dot(r, v), np.dot(r, r) - (coll_rad)**2 
        #discriminant of equation
        D = b**2 - 4*a*c #calculate discriminant
        
        if D < 0 : #no collision (complex solutions)
            t = np.inf 
         
        #collision for container (negative radius)
        elif coll_rad < 0 : 
            time = (-b + np.sqrt(D)) / (2 * a)
            if time > 0:
                t = time 
            else:
                t = np.inf
        
        #avoid problems if balls have overlapped for any reason
        elif np.dot(r, r) < self._rad * 2 : 
            t = np.inf
        
        #choose smallest positive (future) solution
        else: 
            t = np.inf
            times = [(-b + np.sqrt(D))/(2*a), (-b - np.sqrt(D))/(2*a)]
            for time in times:
                if time > 0 and time < t:
                    t = time
        return t
    
    def collide(self, other):
        '''
        Calculates post-collision velocities for the balls taking part in\
        the collision. Does not update the velocities of the objects!

        Parameters
        ----------
        other : Ball object
            The other ball taking part in the collision.

        Returns
        -------
        list
            post-collision velocities of the balls and paths since collision

        '''
        p1 = self._clock * np.linalg.norm(self._vel)
        p2 = other._clock * np.linalg.norm(other._vel) 
        
        rpos = self._pos - other._pos
        rvel = self._vel - other._vel
        
        v1 = self._vel - ((2*other._mass / (self._mass + other._mass)) \
            * (np.dot(rvel, rpos)/np.dot(rpos, rpos))) * rpos
            
        v2 = other._vel + ((2*self._mass / (self._mass + other._mass)) \
           *  (np.dot(rvel, rpos)/np.dot(rpos, rpos))) * rpos
            
        return [[v1, v2], [p1, p2]]
            
    def get_patch(self, btype, color = "r"):
        '''
        Generates patch object for the ball.

        Parameters
        ----------
        btype : kwarg
            specifies whether ball is regular ball or container.
        color : kwarg, optional
            color of fill. The default is "r".

        Returns
        -------
        patch : matplotlib Patch object

        '''
        if btype == "container":
            patch=pl.Circle(self._pos, -self._rad, fill = False, \
                            ec = "b", ls = "solid")
        if btype == "ball":
            patch = pl.Circle(self._pos, self._rad, fc = color, \
                              ec = "k", ls = "solid")
            
        return patch
        
#%%
