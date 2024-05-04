# use optimizedDP for BRT computation for vehicle-vehicle systems 
# also have the worst case 
# * modified BRT that does not do worst case but bounds the disturbance (the action of the other vehicle by their predictive motion instead of worst case)
# practical BRT is the union of all the vehicle-vehicle BRTs for each additional vehicle in the system 
# TODO: Pedestrian-Vehicle BRT
#compute the backwards reachable tube if not computed or call the backwards reachable tube
# color the vehicles yellow if violating worst case BRT and orange if violating predictive BRT
# include a parameter that you can toggle on to record the time it takes to compute the BRT
from abc import ABC
import imp
import numpy as np
#from odp.Grid import Grid
#from odp.Shapes import *
from gymnasium import Wrapper

from highway_env.envs.common.action import Action, ActionType, action_factory
import pickle

# Specify the  file that includes dynamic systems
from highway_env.vehicle.hji_dynamics import HJIVehicle
# Plot options
#from odp.Plots import PlotOptions
#from odp.Plots import plot_isosurface, plot_valuefunction
# Solver core
#from odp.solver import HJSolver, computeSpatDerivArray
from highway_env.envs.common.abstract import AbstractEnv
import math
import os

def define_rel_coord(ego_vehicle, other_vehicle):
    ego_vehicle_heading = int(np.arctan2(ego_vehicle[3],ego_vehicle[2]))
    other_vehicle_heading = int(np.arctan2(other_vehicle[3],other_vehicle[2]))
    ego_vehicle_speed =int(np.sqrt(ego_vehicle[3]**2+ego_vehicle[2]**2))
    other_vehicle_speed = int(np.sqrt(other_vehicle[3]**2+other_vehicle[2]**2))
    x_rel = int(ego_vehicle[0] - other_vehicle[0])
    y_rel = int(ego_vehicle[1] - other_vehicle[1])
    heading_rel = int((other_vehicle_heading - ego_vehicle_heading + 180) % 360 - 180)
    return [x_rel, y_rel, heading_rel, ego_vehicle_speed, other_vehicle_speed]

class BRTCalculator(ABC):
    def __init__(self, env: AbstractEnv, conservative = True, preloaded=[]) -> None:

        if (len(preloaded)>0):
            self.BRT_converged = preloaded
            self.obs_type = "Kinematics"
        else:
            config_boundaries = env.config["observation"]["features_range"]

            # g = Grid(grid_min, grid_max, dims, N, pd)
            grid_min = np.array([config_boundaries["x"][0], config_boundaries["y"][0], 
                            -np.pi/2, config_boundaries["vx"][0],  
                            config_boundaries["vx"][0]]) 
            grid_max = np.array([config_boundaries["x"][1],
                                config_boundaries["y"][1], 
                                np.pi/2, config_boundaries["vx"][1], 
                                config_boundaries["vx"][1]])
            dims = np.array(5)
            N = np.array([40, 40, 40, 40, 40])
            pd = [2]
            g = Grid(grid_min, grid_max, dims, N, pd)
            self.obs_type = env.config["observation"]["type"]
            self.absolute = env.config["observation"]["absolute"]
            self.conservative = conservative
            self.last_obs = {}
            self.dt = 1/env.config["simulation_frequency"]
            #self.features = env.config["observation"]["features"]

            # Failure set
            l_x = CylinderShape(g, [2, 3, 4], np.zeros(5), 4.0) # radius in meters 

            lookback_length = 2.0
            t_step = 0.05

            small_delta = 1e-5
            self.sample_relative_system = HJIVehicle(conservative)
    
            tau = np.arange(start=0, stop=lookback_length + small_delta, step=t_step)

            po = PlotOptions(do_plot=False, plot_type="value", plotDims=[0,1], slicesCut=[39,39,39],
                            save_fig=True, filename="plots/2D_4_valuefunction", interactive_html=True)

            compMethods = { "TargetSetMode": "minVWithV0"}
            #result = HJSolver(self.sample_relative_system, g,l_x, tau, compMethods, po, saveAllTimeSteps=False)
            #np.save('converged_brt.npy', result)

            result = np.load('converged_brt.npy')
            plot_valuefunction(g, result, po)
            # The converged BRT

            self.BRT_converged = result
            self.g = g  

    def check_safety_violation(self, obs):
        # Return an array the size of the number of vehicles where the values are either 0 or 1 
        # 1 for safety violation and 0 for no violation,
        # If observation type is invalid, return -1 
        #at this time, support for Kinematics observation only (order to implement: OccupancyGrid, TTC, Grayscale image )
        # example: [1, 0, 0, 1] if the ego vehicle and the third agent vehicle are in violation 
        if self.obs_type != "Kinematics":
            return -1
        
        vehicle_info = []
        ego_info = []
        other_vehicles = 0
        safety_violation = np.zeros(len(obs)) 

        for i in range(len(obs)):
            if i == 0: 
                ego_info = [obs[i][1], obs[i][2], obs[i][3], obs[i][4]]
            elif obs[i][0] >= 1:
                other_vehicles += 1
                vehicle_info.append(define_rel_coord(ego_info, [obs[i][1], obs[i][2], obs[i][3], obs[i][4]]))

        for i in range(other_vehicles):
            # Index mapping!
            x_ind = int(vehicle_info[i][0]*40/200)
            y_ind = int(vehicle_info[i][1]*40/200)
            heading_ind = int(vehicle_info[i][2]*40/(4*np.pi))

            print(self.BRT_converged[x_ind][y_ind][heading_ind][vehicle_info[i][3]][vehicle_info[i][4]])
            if self.BRT_converged[x_ind][y_ind][heading_ind][vehicle_info[i][3]][vehicle_info[i][4]] <= 20000:
                safety_violation[0] = 1
                safety_violation[i] = 1
        
        return safety_violation

    def determine_safe_action(self, obs):
        # derivatives for each state
        x_derivative = computeSpatDerivArray(self.g, self.BRT_converged, deriv_dim=1, accuracy="low")
        y_derivative = computeSpatDerivArray(self.g, self.BRT_converged, deriv_dim=2, accuracy="low")
        heading_derivative = computeSpatDerivArray(self.g, self.BRT_converged, deriv_dim=3, accuracy="low")
        v_r_derivative = computeSpatDerivArray(self.g, self.BRT_converged, deriv_dim=4, accuracy="low")
        v_h_derivative = computeSpatDerivArray(self.g, self.BRT_converged, deriv_dim=5, accuracy="low")

        if self.obs_type != "Kinematics":
            return -1
        
        vehicle_info = []
        ego_info = []
        other_vehicles = 0
        for i in range(len(obs)):
            if i == 0: 
                ego_info = [obs[i][1], obs[i][2], obs[i][3], obs[i][4]]
            elif obs[i][0] >= 1:
                other_vehicles += 1
                vehicle_info.append(define_rel_coord(ego_info, [obs[i][1], obs[i][2], obs[i][3], obs[i][4]]))

        opt_ctl = []
        for i in range(other_vehicles):
            spat_deriv_vector = [x_derivative[tuple(other_vehicles[i])], y_derivative[tuple(other_vehicles[i])],
                            heading_derivative[tuple(other_vehicles[i])], v_r_derivative[tuple(other_vehicles[i])], v_h_derivative[tuple(other_vehicles[i])]]
            opt_ctl.append(self.sample_relative_system.optCtrl_inPython(spat_deriv_vector))

        return np.mean(opt_ctl, axis=0)

class SafetyWrapper:
    #def __init__(self, env, conservative=True, filter_action = False, precomputed_BRT = [], ):
    def __init__(self, env, conservative=True, filter_action = False, precomputed_BRT = [], precomputed_x_deriv = [],
                  precomputed_y_deriv = [], precomputed_heading_deriv = [], precomputed_v_r_deriv = [], precomputed_v_h_deriv = []): #for dev purposes only
        self.env = env
        if (len(precomputed_BRT) == 0):
            self.brt = BRTCalculator(env, conservative)
        else:
            self.brt = BRTCalculator(env, conservative, precomputed_BRT)
            self.x_deriv = precomputed_x_deriv
            self.y_deriv = precomputed_y_deriv
            self.heading_deriv = precomputed_heading_deriv
            self.v_r_deriv = precomputed_v_r_deriv
            self.v_h_deriv = precomputed_v_h_deriv
    
    def step(self, action: Action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        violation = self.brt.check_safety_violation(obs)
        
        vehicles = self.env.road.vehicles
        vehicle_ind_map = {}

        for j in range(len(vehicles)):
            skip = 0
            for i in range(len(obs)):
                if (vehicles[j].heading < np.arctan2(obs[i][4], obs[i][3]) + 0.1 and vehicles[j].heading > np.arctan2(obs[i][4], obs[i][3])-0.1):
                    vehicle_ind_map[j] = i - skip
                else:
                    skip += 1

        for i in range(len(vehicles)):
            if (i in vehicle_ind_map):
                if (violation[vehicle_ind_map[i]]):
                    vehicles[i].unsafe = True

        return obs, reward, terminated, truncated, info, violation[0]

    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()


