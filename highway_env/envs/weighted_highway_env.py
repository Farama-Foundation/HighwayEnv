from __future__ import annotations

import numpy as np
from typing_extensions import override

from highway_env import utils
from highway_env.envs import HighwayEnv
from highway_env.road.lanes.unweighted_lanes import CircularLane, SineLane, StraightLane
from highway_env.road.lanes.lane_utils import LineType
from highway_env.road.road import Road, WeightedRoadnetwork

class WeightedHighwayEnv(HighwayEnv):
    @override
    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=WeightedRoadnetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30, weight= 1,
            ),
            
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )