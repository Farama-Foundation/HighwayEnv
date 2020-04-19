from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.control import MDPVehicle


class TwoWayEnv(AbstractEnv):
    """
        A risk management task: the agent is driving on a two-way lane with icoming traffic.
        It must balance making progress by overtaking and ensuring safety.

        These conflicting objectives are implemented by a reward signal and a constraint signal,
        in the CMDP/BMDP framework.
    """

    COLLISION_REWARD = 0
    LEFT_LANE_CONSTRAINT = 1
    LEFT_LANE_REWARD = 0.2
    HIGH_VELOCITY_REWARD = 0.8

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            "observation": {
                "type": "TimeToCollision",
                "horizon": 5
            },
        })
        return config

    def step(self, action):
        return super().step(action)

    def _reward(self, action):
        """
            The vehicle is rewarded for driving with high velocity
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)

        reward = self.HIGH_VELOCITY_REWARD * self.vehicle.velocity_index / (self.vehicle.SPEED_COUNT - 1) \
            + self.LEFT_LANE_REWARD * (len(neighbours) - 1 - self.vehicle.target_lane_index[2]) / (len(neighbours) - 1)
        return reward

    def _is_terminal(self):
        """
            The episode is over if the ego vehicle crashed or the time is out.
        """
        return self.vehicle.crashed

    def _cost(self, action):
        """
            The constraint signal is the time spent driving on the opposite lane, and occurence of collisions.
        """
        return float(self.vehicle.crashed) + float(self.vehicle.lane_index[2] == 0)/15

    def reset(self):
        self._make_road()
        self._make_vehicles()
        return super().reset()

    def _make_road(self, length=800):
        """
            Make a road composed of a two-way road.
        :return: the road
        """
        net = RoadNetwork()

        # Lanes
        net.add_lane("a", "b", StraightLane([0, 0], [length, 0],
                                            line_types=[LineType.CONTINUOUS_LINE, LineType.STRIPED]))
        net.add_lane("a", "b", StraightLane([0, StraightLane.DEFAULT_WIDTH], [length, StraightLane.DEFAULT_WIDTH],
                                            line_types=[LineType.NONE, LineType.CONTINUOUS_LINE]))
        net.add_lane("b", "a", StraightLane([length, 0], [0, 0],
                                            line_types=[LineType.NONE, LineType.NONE]))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self):
        """
            Populate a road with several vehicles on the road
        :return: the ego-vehicle
        """
        road = self.road
        ego_vehicle = MDPVehicle(road, road.network.get_lane(("a", "b", 1)).position(30, 0), velocity=30)
        road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for i in range(3):
            self.road.vehicles.append(
                vehicles_type(road,
                              position=road.network.get_lane(("a", "b", 1))
                              .position(70+40*i + 10*self.np_random.randn(), 0),
                              heading=road.network.get_lane(("a", "b", 1)).heading_at(70+40*i),
                              velocity=24 + 2*self.np_random.randn(),
                              enable_lane_change=False)
            )
        for i in range(2):
            v = vehicles_type(road,
                              position=road.network.get_lane(("b", "a", 0))
                              .position(200+100*i + 10*self.np_random.randn(), 0),
                              heading=road.network.get_lane(("b", "a", 0)).heading_at(200+100*i),
                              velocity=20 + 5*self.np_random.randn(),
                              enable_lane_change=False)
            v.target_lane_index = ("b", "a", 0)
            self.road.vehicles.append(v)


register(
    id='two-way-v0',
    entry_point='highway_env.envs:TwoWayEnv',
    max_episode_steps=15
)
