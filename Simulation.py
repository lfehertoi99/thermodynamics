# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:42:51 2020

@author: lilif
"""
#%%
import numpy as np
from Ball import Ball
import pylab as pl
from itertools import combinations
import matplotlib.pyplot as plt
#%%
class Simulation:
    '''
    Simulation class to call relevant methods on multiple balls and container \
        which are part of the simulation. Manages observables such as \
            temperature, pressure and ball velocities and obtains plots from \
                the data.
    --------------------------------------------------------------------------
        - n_balls: number of balls in container, int
        - density: (area) density of balls in container, float
        - ball_rad: radius of balls, float
        - ball_mass: mass of balls, float
        - mom: total momentum imparted to container (used to calculate \
                                                     pressure), float
        - time: total time since beginning of simulation, float
        - temp: temperature (ke) of balls 
    --------------------------------------------------------------------------
    Methods: 
        - next_collision (versions 1 and 2)
        - dist_hist
        - pair_dist_hist
        - vel_hist
        - run (versions 1 and 2)
        - sys_kin_e
        Note: for the next_collision and run methods, version 2 has faster
            runtime. 
    -------------------------------------------------------------------------- 
    '''
    
    def __init__(self, n_balls, density = 0.15, ball_rad = 0.5, ball_mass = 1,\
                 mom = 0, time = 0, temp = 1):
        '''
        Parameters
        ----------
        n_balls : int
            number of balls in container.
        density : float, optional
            total area of balls / total area of container. The default is 0.5.
        ball_rad : float, optional
            radius of balls. The default is 0.15.
        ball_mass : float, optional
            mass of balls. The default is 1.
        mom : float, optional
            total momentum imparted to container. The default is 0.
        time : float, optional
            total time since beginning of simulation. The default is 0.

        Returns
        -------
        None.

        '''
        
        self._n_balls = n_balls
        self._density = density
        self._ball_mass = ball_mass
        self._ball_rad = ball_rad
        
        r = np.sqrt((n_balls * ball_rad**2) / density)
        self._container = Ball(np.array([0, 0]), np.array([0, 0]),\
                               rad = -r, mass = 1e40) #negative radius!
        
        #square container of size np.floor(np.sqrt(n_balls)) + 1
        #arrange balls evenly in grid
        
        a = 3 * self._ball_rad * np.ceil(np.sqrt(n_balls))
        x =  y = np.arange(-a/2 + ball_rad, a/2 + ball_rad, 3*ball_rad)
        xx,yy = np.meshgrid(x,y)
        coords = np.array((xx.ravel(), yy.ravel())).T
        
        if len(coords) < self._n_balls or a/2 > r:
            raise Exception("Density too high")
        
        np.random.seed(0) #for reproducibility
        
        #initialize velocities randomly
        self._balls = [Ball(coords[i], \
                      np.sqrt(temp)*np.random.normal(size = 2), \
                      rad = self._ball_rad, mass = self._ball_mass) \
                      for i in range(n_balls)]
        
        #total momentum imparted to container and time since beginning of simulation
        self._mom = mom
        self._time = time
        
        self._velocities = []
        self._paths = []
        

    def next_collision1(self):
        '''
        Calculates time to next collision. Evolves system forward by that time.
        Calculates post-collision velocities and updates velocities of balls
        involved. Updates momentum imparted to container and time since
        beginning of simulation. 
        '''
        k = combinations(self._balls, 2) #distinct pairs of balls
        v = np.around(np.array([b1.time_to_collision(b2) for (b1, b2) in k] + \
                  [b.time_to_collision(self._container) for b in self._balls])\
                      , 4)

            
        l = [x for x in combinations(self._balls, 2)] + \
            list(map(lambda e: (e, self._container), self._balls))
        
        times = dict(zip(l, v)) #put pairs: time into dictionary
        #select smallest time and extract balls involved
        c_balls, c_time = sorted(times.items(), key = lambda item: item[1])[0]
        
        #move all balls to next frame:
        for b in self._balls:
            b.move(c_time)
            b._clock += c_time

        #calculate post-coll. velocities:
        b1, b2 = c_balls[0], c_balls[1]
        P = b1.collide(b2)
        pc_vel, paths = P[0], P[1]
        
        for p in paths:
            self._paths.append(p)
        
        #update velocities:
        b1.setvel(pc_vel[0])
        b2.setvel(pc_vel[1])
        
        b1._clock = 0
        b2._clock = 0

        #set container position and velocity to zero to avoid problems
        self._container.setpos(np.array([0.0, 0.0]))
        self._container.setvel(np.array([0.0, 0.0]))

        #move balls along slightly to avoid problems 
        for b in self._balls:
            b.move(0.00001)
        
        #update time and total momentum transfer to container
        self._time += c_time
        if self._container == b2:
            self._mom += 2 * self.ball_mass * np.linalg.norm(b1.getvel())
            
    def next_collision2(self, prev_coll, times):
        '''
        Calculates time to next collision. Evolves system forward by that time.
        Calculates post-collision velocities and updates velocities of balls
        involved. Updates momentum imparted to container and time since
        beginning of simulation. 
        '''
        if not prev_coll:
            k = combinations(self._balls, 2) #distinct pairs of balls
            v = np.around(np.array([b1.time_to_collision(b2) for (b1, b2) \
                                    in k] +
                      [b.time_to_collision(self._container) for b in \
                       self._balls]), 4)


            l = [x for x in combinations(self._balls, 2)] + \
                list(map(lambda e: (e, self._container), self._balls))
        
            times = dict(zip(l, v)) #put pairs: time into dictionary
            
        else:
            b1, b2 = prev_coll[0], prev_coll[1]
            b1_pairs = [(b1, b) for b in self._balls if b not in prev_coll]+\
                [(b1, self._container)]
            b2_pairs = [(b2, b) for b in self._balls if b not in prev_coll]+\
                [(b2, self._container)]
            
            for x in b1_pairs:
                if x in times:
                    times[x] = x[0].time_to_collision(x[1])
                elif (x[-1], x[0]) in times:
                    times[(x[-1], x[0])] = x[0].time_to_collision(x[1])
                    
            for x in b2_pairs:
                if x in times:
                    times[x] = x[0].time_to_collision(x[1])
                elif (x[-1], x[0]) in times:
                    times[(x[-1], x[0])] = x[0].time_to_collision(x[1])
            
            if (b1, b2) in times:
                times[b1, b2] = b1.time_to_collision(b2)
            elif (b2, b1) in times:
                times[b2, b1] = b1.time_to_colision(b2)
                
        #select smallest time and extract balls involved
        c_balls, c_time = sorted(times.items(), key = lambda item: item[1])[0]

        #move all balls to next frame:
        for b in self._balls:
            b.move(c_time)
            b._clock += c_time 
            
        for k in times.keys():
            times[k] -= np.around(c_time, 6)
            
        #calculate post-coll. velocities:
        b1, b2 = c_balls[0], c_balls[1]
        P = b1.collide(b2)
        
        pc_vel, paths = P[0], P[1]

        for p in paths:
         self._paths.append(p)
        
        b1._clock = 0
        b2._clock = 0
        
        #update velocities:
        b1.setvel(pc_vel[0])
        b2.setvel(pc_vel[1])
        
        #set container position and velocity to zero to avoid problems
        self._container.setpos(np.array([0.0, 0.0]))
        self._container.setvel(np.array([0.0, 0.0]))

        #move balls along slightly to avoid problems 
        for b in self._balls:
            b.move(0.00001)
        
        #update time and total momentum transfer to container
        self._time += c_time
        if self._container == b2:
            self._mom += 2 * self._ball_mass * np.linalg.norm(b1.getvel())
            
        return [b1, b2], times
    
    def dist_hist(self, bins = 10): #distance of balls from centre
        '''
        Returns plot (histogram) of distribution of distance of balls from
        centre. Expect number of balls in each bin to vary linearly with
        radius when the number of balls is large. The plot can be used as a 
        debugging aid to make sure the system is behaving as expected.
        
        Parameters
        -------
        bins: int
            number of bins to use in histogram. Default is 10.
        Returns
        -------
        None
        '''
        
        x = [np.linalg.norm(p) for p in [b.getpos() for b in self._balls]]
        
        fig, ax = plt.subplots()
        ax.set_title("Distance of balls from centre")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Ball count")
        ax.hist(x, bins = bins)
        
        plt.show()
    
    def pair_dist_hist(self, bins = 10): #distance of balls from each other
        '''
        Returns plot (histogram) of distribution of distance of pairs of balls
        from each other. This should resemble a Gaussian. The plot can be used
        as a debugging aid to make sure the system is behaving as expected.
        
        Parameters
        -------
        bins: int
            number of bins to use in histogram. Default is 10.
        Returns
        -------
        None
        '''
        pairs = combinations(self._balls, 2)
        pair_dists = [np.linalg.norm([a[0].getpos() - a[1].getpos()]) for \
                      a in pairs]
        
        fig, ax = plt.subplots()
        ax.set_title("Distance between pairs of balls")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Pair count")
        ax.hist(pair_dists, bins = bins)
        
        plt.show()
        
    def vel_hist(self, bins = 10): #instantaneous velocity distribution
        '''
        Returns plot (histogram) of velocity distribution of balls in current
        frame. This should look like the Maxwell-Boltzmann distribution for a
        large number of balls. The plot can be used as a debugging aid to 
        make sure the system is behaving as expected.
        
        Parameters
        -------
        bins: int
            number of bins to use in histogram. Default is 10.
        Returns
        -------
        None
        '''
        
        x = [np.linalg.norm(v) for v in [b.getvel() for b in self._balls]]
        
        fig, ax = plt.subplots()
        ax.set_title("Velocity distribution")
        ax.set_xlabel("Velocity")
        ax.set_ylabel("Ball count")
        ax.hist(x, bins = bins)
        
        plt.show()
        
                
    def sys_kin_e(self):
        '''
        Obtain the total kinetic energy of the system. Calls the kin_e method
        of the Ball class for every ball and adds results. Kinetic energy is
        calculated as 0.5*mass*(velocity**2) for each ball.

        Returns
        -------
        float
            total kinetic energy of system

        '''
        return sum([b.kin_e() for b in self._balls]) + self._container.kin_e()
    
    def sys_mom(self):
        '''
        Obtain the total momentum of the balls in the system. The container
        momentum should be zero, since the velocity is manually set to zero 
        after each collision. 

        Returns
        -------
        float
            total momentum of balls in system.

        '''
        x = np.array([b.getvel()[0] for b in self._balls])
        y = np.array([b.getvel()[1] for b in self._balls])
        
        x_tot = np.sum(x)
        y_tot = np.sum(y)
        
        return self._ball_mass * np.linalg.norm(np.array([x_tot, y_tot]))
    
    
    def run1(self, num_frames, animate = False, plots = False):
            
        '''
        Runs the system through the required number of frames and collects
        data on the macroscopic state of the system during the simulation. 
        Has option to create a visual representation of the system as well
        as plots to characterize the system's time evolution.

        Parameters
        ----------
        num_frames : int
            DESCRIPTION.
        animate : bool, optional
            An animation of the system is generated if True. The default 
            is False.
        plots: 
            Plots of temperature evolution, pressure evolution and velocity
            distribution are generated if True. The default is False.

        Returns
        -------
        list
            list containing mean pressure from past 50 frames and temperature
            of last frame (which should be constant through all frames)

        '''
    
        #keep track of time, pressure and temperature
        vel = []
        time = []
        pressure = [] 
        temp = []
        
        #run entire simulation
        for frame in range(num_frames):
        
            if animate:
                f = pl.figure()
                #add axes and artist
                ax = pl.axes(xlim=(-self._container._rad, \
                                   self._container._rad),\
                             ylim=(-self._container._rad, \
                                   self._container._rad), aspect = "1")
                    
                ax.add_artist(self._container.get_patch(btype = \
                                                        "container"))
                #plot balls in their current positions
                for b in self._balls:
                    ax.add_patch(b.get_patch(btype = "ball"))
                #delay    
                pl.pause(0.01)
            
            #perform collision, update time, temperature and pressure
            self.next_collision1()
            time.append(self._time)
            pressure.append((self._mom / self._time)/ \
                            (-self._container._rad))
            #choose units such that k_B = 1. DOF = 2.
            temp.append((self.sys_kin_e() / self.n_balls))
            v = [np.linalg.norm(b.getvel()) for b in self._balls]
            #vel += v
            self._velocities += v
        
        if plots:
            #PLOT TEMPERATURE AND PRESSURE EVOLUTION AT END
            fig, ax1 = plt.subplots()
            ax1.set_title("Temperature of gas in container")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Temperature")
            ax1.plot(time, temp)
            plt.show()
            
            fig, ax2 = plt.subplots()
            ax2.set_title("Pressure on side of container")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Pressure")
            ax2.plot(time, pressure)
            plt.show()
            
            fig, ax3 = plt.subplots()
            ax3.set_title("Velocity distribution")
            ax3.set_xlabel("Velocity")
            ax3.set_ylabel("Frequency")
            ax3.hist(self._velocities, bins = 15, density = True)
            plt.show()
            
        return [np.mean(pressure[-50:-1]), temp[-1]] #return final values
    
    def run2(self, num_frames, animate = False, plots = False):
        '''
        Runs the system through the required number of frames and collects
        data on the macroscopic state of the system during the simulation. 
        Has option to create a visual representation of the system as well
        as plots to characterize the system's time evolution.

        Parameters
        ----------
        num_frames : int
            DESCRIPTION.
        animate : bool, optional
            An animation of the system is generated if True. The default 
            is False.
        plots: 
            Plots of temperature evolution, pressure evolution and velocity
            distribution are generated if True. The default is False.

        Returns
        -------
        list
            list containing mean pressure from past 50 frames and temperature
            of last frame (which should be constant through all frames)

        '''
        
        #keep track of time, pressure and temperature
        vel = []
        time = []
        pressure = [] 
        temp = []
        mom = []
        
        prev_coll = None
        times = None
        
        #run entire simulation
        for frame in range(num_frames):
            
            if animate:
                f = pl.figure()
                #add axes and artist
                ax = pl.axes(xlim=(-self._container._rad, \
                                   self._container._rad), \
                             ylim=(-self._container._rad, \
                                   self._container._rad), aspect = "1")
                    
                ax.add_artist(self._container.get_patch(btype = "container"))
                #plot balls in their current positions
                for b in self._balls:
                    ax.add_patch(b.get_patch(btype = "ball"))
                #delay    
                pl.pause(0.01)
            
            #perform collision, update time, temperature and pressure
            prev_coll, times = self.next_collision2(prev_coll, times)
            time.append(self._time)
            pressure.append((self._mom / self._time))
            #choose units such that k_B = 1. DOF = 2.
            temp.append((self.sys_kin_e() / self._n_balls))
            mom.append(self.sys_mom())
            v = [np.linalg.norm(b.getvel()) for b in self._balls]
            #vel += v
            self._velocities += v
        
        if plots:
            #PLOT TEMPERATURE AND PRESSURE EVOLUTION AT END
            fig, ax1 = plt.subplots(figsize = (10, 6))
            ax1.set_title("Temperature of gas in container")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Temperature")
            ax1.plot(time, temp)
            plt.show()
            
            fig, ax2 = plt.subplots()
            ax2.set_title("Pressure on side of container")
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Pressure")
            ax2.plot(time, pressure)
            plt.show()
            
            fig, ax3 = plt.subplots()
            ax3.set_title("Velocity distribution")
            ax3.set_xlabel("Velocity")
            ax3.set_ylabel("Frequency")
            ax3.hist(self._velocities, bins = 15, density = True)
            plt.show()
            
            fig, ax4 = plt.subplots()
            ax4.set_title("Total momentum over time")
            ax4.set_xlabel("Time")
            ax4.set_ylabel("Total momentum")
            ax4.plot(time, mom)
            
        return [np.mean(pressure[-50:-1]), temp[-1]] #return final values only
