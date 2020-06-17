import gym

import highway_env


def test_preprocessors():
    env = gym.make('highway-v0')
    env = env.simplify()
    env = env.change_vehicles("highway_env.vehicle.behavior.IDMVehicle")
    env = env.set_preferred_lane(0)
    env = env.set_route_at_intersection("random")
    env = env.set_vehicle_field(("crashed", False))
    env = env.call_vehicle_method(("plan_route_to", "1"))
    env = env.randomize_behaviour()

    env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, _, _ = env.step(action)
    env.close()

    assert env.observation_space.contains(obs)
    assert 0 <= reward <= 1

