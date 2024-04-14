# use optimizedDP for BRT computation for vehicle-vehicle systems 
# also have the worst case 
# * modified BRT that does not do worst case but bounds the disturbance (the action of the other vehicle by their predictive motion instead of worst case)
# practical BRT is the union of all the vehicle-vehicle BRTs for each additional vehicle in the system 
# TODO: Pedestrian-Vehicle BRT
#compute the backwards reachable tube if not computed or call the backwards reachable tube
# color the vehicles yellow if violating worst case BRT and orange if violating predictive BRT
# include a parameter that you can toggle on to record the time it takes to compute the BRT

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
from highway_env.vehicle.dynamics import BicycleVehicle
import math
import os