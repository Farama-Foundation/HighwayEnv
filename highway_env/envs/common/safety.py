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
from odp.Grid import Grid
from odp.Shapes import *
import pickle

# Specify the  file that includes dynamic systems
from highway_env.vehicle.hji_dynamics import HJIVehicle
# Plot options
from odp.Plots import PlotOptions
from odp.Plots import plot_isosurface, plot_valuefunction
# Solver core
from odp.solver import HJSolver, computeSpatDerivArray
from highway_env.envs.common.abstract import AbstractEnv
import math
import os

def define_rel_coord(ego_vehicle, other_vehicle):
    ego_vehicle_heading = np.arctan2(ego_vehicle[3]/ego_vehicle[2])
    other_vehicle_heading = np.arctan2(other_vehicle[3]/other_vehicle[2])
    ego_vehicle_speed = np.sqrt(ego_vehicle[3]**2+ego_vehicle[2]**2)
    other_vehicle_speed = np.sqrt(other_vehicle[3]**2+other_vehicle[2]**2)
    x_rel = ego_vehicle[0] - other_vehicle[0]
    y_rel = ego_vehicle[1] - other_vehicle[1]
    heading_rel = (other_vehicle_heading - ego_vehicle_heading + 180) % 360 - 180
    return [x_rel, y_rel, heading_rel, ego_vehicle_speed, other_vehicle_speed]

class BRTCalculator(ABC):
    def __init__(self, env: AbstractEnv, conservative = True) -> None:
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
        print(grid_min)
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
        #po = PlotOptions(do_plot=True, plot_type="value", plotDims=[0,1],
        #          slicesCut=[19, 30])
        po = PlotOptions(do_plot=False, plot_type="value", plotDims=[0,1], slicesCut=[39,39,39],
                         save_fig=True, filename="plots/2D_4_valuefunction", interactive_html=True)

        compMethods = { "TargetSetMode": "minVWithV0"}
        #result = HJSolver(self.sample_relative_system, g,l_x, tau, compMethods, po, saveAllTimeSteps=False)
        #np.save('converged_brt.npy', result)

        result = np.load('converged_brt.npy')
        plot_valuefunction(g, result, po)
        # The converged BRT
        last_time_step_result = result[..., 0]

        self.BRT_converged = last_time_step_result
        self.g = g

    def check_safety_violation(self, obs):
        # Return 1 for violation and 0 for no violation, -1 for invalid observation type
        #at this time, support for Kinematics observation only (order to implement: OccupancyGrid, TTC, Grayscale image )
        if self.obs_type != "Kinematics":
            return -1
        
        vehicle_info = []
        ego_info = []
        other_vehicles = 0

        x_derivative = computeSpatDerivArray(self.g, self.BRT_converged, deriv_dim=1, accuracy="low")
        y_derivative = computeSpatDerivArray(self.g, self.BRT_converged, deriv_dim=2, accuracy="low")
        heading_derivative = computeSpatDerivArray(self.g, self.BRT_converged, deriv_dim=3, accuracy="low")
        v_r_derivative = computeSpatDerivArray(self.g, self.BRT_converged, deriv_dim=4, accuracy="low")
        v_h_derivative = computeSpatDerivArray(self.g, self.BRT_converged, deriv_dim=5, accuracy="low")

        for i in range(len(obs)):
            if i == 0: 
                ego_info = [obs[i][1], obs[i][2], obs[i][3], obs[i][4]]
            elif obs[i][0] >= 1:
                other_vehicles += 1
                vehicle_info.append(define_rel_coord(ego_info, [obs[i][1], obs[i][2], obs[i][3], obs[i][4]]))

        safety_violation = 0 
        for i in range(other_vehicles):
            if self.BRT_converged[vehicle_info[i][0]][vehicle_info[i][1]][vehicle_info[i][2]][vehicle_info[i][3]][vehicle_info[i][4]] <= 0:
               # if not self.conservative and self.last_obs != {}:
                #    old_action = [self.last_obs[],]
                #else:
                safety_violation = 1
        
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
