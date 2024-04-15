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

# Specify the  file that includes dynamic systems
from odp.dynamics import DubinsCapture, Plane2D, Plane1D
# Plot options
from odp.Plots import PlotOptions
from odp.Plots import plot_isosurface, plot_valuefunction
# Solver core
from odp.solver import HJSolver, computeSpatDerivArray
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.vehicle.dynamics import BicycleVehicle
import math
import os

class BRTCalculator(ABC):
    def __init__(self, env: AbstractEnv) -> None:
        config_boundaries = env.config["observation"]["features_range"]
        g = Grid(np.array([config_boundaries["x"][0], config_boundaries["y"][0], config_boundaries[3][0], 0]), np.array([config_boundaries["x"][1], config_boundaries["y"][1], config_boundaries[3][1], 2*np.pi]), [4], [40, 40, 40, 40], [])
    
