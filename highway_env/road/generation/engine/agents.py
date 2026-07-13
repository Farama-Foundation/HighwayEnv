from collections import defaultdict

import numpy as np
from noise import pnoise2

from ..spatial_hash import get_proximal_lanes_wrt_gridpoint, point_to_gridpoint
from .gen_utils import Lane, tqdm


def generate_road_network_skeleton(
    target_num_endpoints: int,
    forward_speed: int,
    merge_radius: int,
    prevent_replication_radius: int,
    age_of_maturity: int,
    perlin_variation_params: dict,
    disable_prints: bool = False,
    rng: np.random.Generator | None = None,
) -> list[Lane]:
    """
    Uses a swarm of moving, replicating agents to sketch out a rough draft of
    the road topology.

    :param target_num_endpoints: number of intersections to generate.
    :param forward_speed: distance an agent travels every timestep
    :param merge_radius: minimum distance an agent can be from another
    agent/lane before being killed (merged)
    :param prevent_replication_radius: distance an agent has to be from
    another agent/lane in order to be able to replicate
    :param age_of_maturity: number of timesteps before an agent can
    replicate
    :param perlin_variation_params: determines the rates of turning and
    replication and how they vary over physical space
    :param disable_prints: disables progress and status printing
    :param rng: random number generator
    :return: list of lanes
    """
    if rng is None:
        rng = np.random.default_rng()

    for feature_params_dict in perlin_variation_params.values():
        feature_params_dict["x"] = rng.integers(0, 10000)
        feature_params_dict["y"] = rng.integers(0, 10000)

    class PerlinVariation:
        scale = 200
        octaves = 1
        persistence = 0.1
        lacunarity = 2.0

        @staticmethod
        def paramAt(param, pos):
            x = perlin_variation_params[param]["x"]
            y = perlin_variation_params[param]["y"]
            upper = perlin_variation_params[param]["upper"]
            lower = perlin_variation_params[param]["lower"]
            noise_val = pnoise2(
                (pos[0] / PerlinVariation.scale) + x,
                (pos[1] / PerlinVariation.scale) + y,
                octaves=PerlinVariation.octaves,
                persistence=PerlinVariation.persistence,
                lacunarity=PerlinVariation.lacunarity,
            )
            return (
                ((upper - lower) * noise_val * abs(noise_val)) + upper + lower
            ) / 2.0

    class ConstructionAgent:
        fork_angles = [-np.pi / 2, 0, np.pi / 2]

        fork_possibilities = [
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]

        @classmethod
        def random_fork_config(cls):
            return ConstructionAgent.fork_possibilities[
                rng.integers(0, len(ConstructionAgent.fork_possibilities))
            ]

        def __init__(
            self,
            start_location,
            position=None,
            orientation=0,
            angular_velocity=0,
        ):
            if position is None:
                self.position = np.zeros(2)
            else:
                self.position = position

            self.start_location = str(start_location)
            self.end_location = str(-1)
            self.orientation = orientation
            self.angular_velocity = angular_velocity
            self.history = []

        def step(self):
            jitteriness = PerlinVariation.paramAt("jitteriness", self.position)
            max_turn_speed = PerlinVariation.paramAt("max_turn_speed", self.position)

            turn_friction = 1 - jitteriness
            turn_acceleration_range = max_turn_speed * (1 / turn_friction - 1)

            self.position += (
                np.array([np.cos(self.orientation), np.sin(self.orientation)])
                * forward_speed
            )

            self.angular_velocity += turn_acceleration_range * (rng.random() - 0.5)
            self.angular_velocity *= turn_friction
            self.orientation += self.angular_velocity

            self.history.append(self.position.copy())

    spatial_hash_gridsize = max(50, prevent_replication_radius)

    successful_generation = False
    MAX_GENERATION_ATTEMPTS = 10
    for generation_attempt in range(MAX_GENERATION_ATTEMPTS):
        lanes: list[Lane] = []
        grid_to_lanes = defaultdict(set)
        agents = [ConstructionAgent(start_location=0)]
        num_locations = 1
        simulation_step = 0
        with tqdm(
            total=target_num_endpoints,
            disabled=disable_prints,
            desc="Generating road network skeleton",
        ) as pbar:
            while num_locations < target_num_endpoints and len(agents) > 0:
                for agent in agents:
                    agent.step()

                agents_to_remove = []
                agents_to_add = []
                for agent in agents:
                    if len(agent.history) <= age_of_maturity:
                        continue  # prevent replication/merging

                    # Death due to merging or spontaneous death chance
                    prevent_replication = False
                    merge_enacted = False

                    for other_agent in agents:
                        for i, position in enumerate(other_agent.history):
                            if (
                                agent is other_agent
                                and i >= len(agent.history) - age_of_maturity
                            ):
                                break

                            dist = np.linalg.norm(position - agent.position)
                            if (
                                agent is not other_agent
                                and dist < prevent_replication_radius
                            ):
                                prevent_replication = True
                            if dist < merge_radius:
                                merge_enacted = True
                                break

                    if not merge_enacted:
                        gridpoint = point_to_gridpoint(
                            agent.position, spatial_hash_gridsize
                        )
                        proximal_lanes = get_proximal_lanes_wrt_gridpoint(
                            grid_to_lanes, gridpoint, extended=True
                        )
                        for laneID in proximal_lanes:
                            lane = lanes[laneID]
                            for i, position in enumerate(lane.points):
                                dist = np.linalg.norm(position - agent.position)
                                if dist < prevent_replication_radius:
                                    prevent_replication = True
                                if dist < merge_radius:
                                    merge_enacted = True
                                    break

                    true_population = len(agents) - len(agents_to_remove)
                    spontaneous_death_chance = PerlinVariation.paramAt(
                        "spontaneous_death_chance", agent.position
                    )

                    if merge_enacted or (
                        true_population > 3 and rng.random() < spontaneous_death_chance
                    ):
                        agents_to_remove.append(agent)
                        agent.end_location = str(num_locations)
                        num_locations += 1
                        continue

                    if prevent_replication:
                        continue

                    # Replication
                    replication_chance = PerlinVariation.paramAt(
                        "replication_chance", agent.position
                    )
                    if rng.random() < replication_chance:
                        agent.end_location = str(num_locations)
                        agents_to_remove.append(agent)

                        fork_config = ConstructionAgent.random_fork_config()
                        for i, angle in enumerate(ConstructionAgent.fork_angles):
                            if fork_config[i] == 1 or true_population < 3:
                                new_agent = ConstructionAgent(
                                    start_location=num_locations,
                                    position=agent.position.copy(),
                                    orientation=agent.orientation + angle,
                                )

                                agents_to_add.append(new_agent)

                        num_locations += 1

                for dying_agent in agents_to_remove:
                    # Agent path history gets turned into an official Lane
                    new_lane = Lane(
                        start=dying_agent.start_location,
                        end=dying_agent.end_location,
                        points=dying_agent.history[:-1],
                    )

                    for point in new_lane.points:
                        gridpoint = point_to_gridpoint(point, spatial_hash_gridsize)
                        grid_to_lanes[gridpoint].add(len(lanes))

                    lanes.append(new_lane)
                    agents.remove(dying_agent)

                for new_agent in agents_to_add:
                    agents.append(new_agent)

                simulation_step += 1
                pbar.n = min(num_locations, target_num_endpoints)
                pbar.refresh()

        for agent in agents:
            new_lane = Lane(
                start=agent.start_location,
                end=str(num_locations),
                points=agent.history,
            )
            lanes.append(new_lane)
            num_locations += 1

        if num_locations >= target_num_endpoints:
            successful_generation = True
            break
        elif not disable_prints:
            print(
                "Swarm generation fault"
                " [insufficient number of nodes reached];"
                " restarting"
            )

    if not successful_generation:
        raise TimeoutError(
            "Generation swarm keeps dying out;"
            " try increasing the replication rate"
            " or decreasing the turn speed."
        )

    return lanes
