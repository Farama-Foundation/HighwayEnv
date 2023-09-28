import torch
import math
import numpy as np
import copy

class Agent:
    def __init__(self) -> None:
        pass
    def select_action(self):
        pass
        
class VictimAgent(Agent):
    def __init__(self, action_selection_model, device) -> None:
        super().__init__()
        self.device = device
        # the model object and parameter should be loaded before the victim agent is initiated
        self.select_action_model = action_selection_model

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).view(25).unsqueeze(0)
        return self.select_action_model(state).max(1)[1].view(1, 1).item()

    
class AttackerAgent(Agent):
    def __init__(self, id) -> None:
        super().__init__()
        self.id = id
        # the model object and parameter should be loaded before the victim agent is initiated
        # self.select_action_model = action_selection_model

    def select_action(self, a):
        pass
        # return self.select_action_model(state)

class PerfectVictim(Agent):
    def __init__(self, env) -> None:
        super().__init__()
        self.env = env
        self.victim_action_range = [0,1,2,3,4]
        self.actions = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }

    def select_action(self, attacker_actions):
        frames = int(self.env.config["simulation_frequency"] // self.env.config["policy_frequency"])
        dt = 1 / self.env.config["simulation_frequency"]

        # each controllable vehicles and the victim save their previous states
        saved_victim_state = self.save_vehicle_state(self.env.victim)
        saved_vehicle_states = []
        for i, vehicle in enumerate(self.env.controlled_vehicles):
            saved_vehicle_states.append(self.save_vehicle_state(vehicle))

        forbidden_actions = set()
        for victim_a in self.victim_action_range:
            if len(forbidden_actions) == len(self.victim_action_range):
                break
            if victim_a not in forbidden_actions:
                # print("victim action sim: ", victim_a)
                victim_crashed = False
                for i, vehicle in enumerate(self.env.controlled_vehicles):
                    self.load_vehicle_state(vehicle, saved_vehicle_states[i])
                self.load_vehicle_state(self.env.victim, saved_victim_state)
                action_passed = False
                for frame in range(frames):
                    # print("xxxxxxxxxxxxx ", self.env.controlled_vehicles[2].crashed)
                    # print("yyyyyyyyyyyyyy ", self.env.controlled_vehicles[2].impact)
                    # print("saved_vehicle_states[2][impact]", saved_vehicle_states[2]["impact"])
                    # TODO: victim update state using victim_a, attackers update their state based on attacker_actions
                    if not action_passed:
                        for i, vehicle in enumerate(self.env.controlled_vehicles):
                            if vehicle.crashed or vehicle.destroyed:
                                continue
                            vehicle.act(self.actions[attacker_actions[i]])
                            self.step(vehicle, dt)
                        self.env.victim.act(self.actions[victim_a])
                        self.step(self.env.victim, dt)
                        action_passed = True
                    else:
                        for i, vehicle in enumerate(self.env.controlled_vehicles):
                            if vehicle.crashed or vehicle.destroyed:
                                continue
                            vehicle.act()
                            self.step(vehicle, dt)
                        self.env.victim.act()
                        self.step(self.env.victim, dt)
                    # print("victim position: ", self.env.victim.position)
                    # print("yoooooooooo ", self.env.controlled_vehicles[2].crashed)
                    # check collision
                    for i, vehicle in enumerate(self.env.controlled_vehicles):
                        # print("attacker ", str(i), " position: ", vehicle.position)
                        # print("dt: ", dt)
                        if vehicle.crashed or vehicle.destroyed:
                            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^: ", str(vehicle.destroyed), str(vehicle.crashed))
                            # print("saved_vehicle_states[2][crashed]", saved_vehicle_states[2]["crashed"])
                            continue
                        if self.victim_handle_collisions(vehicle, dt):
                            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
                            forbidden_actions.add(victim_a)
                            victim_crashed = True
                            # self.load_vehicle_state(self.env.victim, ori_saved_victim_state)
                            break
                    if victim_crashed:
                        break
                    for i, vehicle in enumerate(self.env.controlled_vehicles):
                        for other in self.env.controlled_vehicles[i+1:]:
                            vehicle.handle_collisions(other, dt)
                        # for other in self.objects:
                        #     vehicle.handle_collisions(other, dt)
            else:
                continue
        # load controllable vehicle states
        for i, vehicle in enumerate(self.env.controlled_vehicles):
            self.load_vehicle_state(vehicle, saved_vehicle_states[i])
        self.load_vehicle_state(self.env.victim, saved_victim_state)
        # print("victim speed : {}".format(self.env.victim.speed))
        # select victim action based on forbidden_actions
        print(forbidden_actions)
        victim_l_index = self.env.victim.lane_index
        for i in range(len(self.env.controlled_vehicles)):
            other_l_index = self.env.controlled_vehicles[i].lane_index
            if other_l_index[2] == victim_l_index[2]:
                # attacker is the same lane as the victim
                victim_local_x = self.env.road.network.get_lane(victim_l_index).local_coordinates(self.env.victim.position)[0]
                attacker_local_x = self.env.road.network.get_lane(victim_l_index).local_coordinates(self.env.controlled_vehicles[i].position)[0]
                print(" distance: {}".format(attacker_local_x - victim_local_x))
                if attacker_local_x > victim_local_x and (attacker_local_x - victim_local_x) <= self.env.config["close_vehicle_threshold"]+10:
                    # attacker ahead
                    
                    if self.env.controlled_vehicles[i].speed < self.env.victim.speed and 4 not in forbidden_actions:
                        # print("action 4")
                        return 4
                    if self.env.controlled_vehicles[i].speed > self.env.victim.speed and 3 not in forbidden_actions:
                        # print("action 3")
                        return 3
                if attacker_local_x < victim_local_x and (victim_local_x - attacker_local_x) <= self.env.config["close_vehicle_threshold"]+10:
                    #attacker behind
                    if self.env.controlled_vehicles[i].speed < self.env.victim.speed and 4 not in forbidden_actions:
                        return 4
                    if self.env.controlled_vehicles[i].speed > self.env.victim.speed and 3 not in forbidden_actions:
                        return 3

        if len(forbidden_actions) == len(self.victim_action_range):
            return 1 # IDLE
        if 3 not in forbidden_actions:
            return 3 # FASTER
        if 2 not in forbidden_actions:
            return 2 # LANERIGHT
        if 1 not in forbidden_actions:
            return 1 # IDLE
        if 4 not in forbidden_actions:
            return 4 # SLOWER
        if 0 not in forbidden_actions:
            return 0 # LANELEFT
        
    
        
    def save_vehicle_state(self, vehicle):
        state = {}
        state["heading"] = copy.deepcopy(vehicle.heading)
        state["position"] = copy.deepcopy(vehicle.position)
        state["speed"] = copy.deepcopy(vehicle.speed)
        state["target_speed"] = copy.deepcopy(vehicle.target_speed)
        state["target_lane_index"] = copy.deepcopy(vehicle.target_lane_index)
        state["speed_index"] = copy.deepcopy(vehicle.speed_index)
        state["crashed"] = copy.deepcopy(vehicle.crashed)
        state["collide_with"] = copy.deepcopy(vehicle.collide_with)
        state["impact"] = copy.deepcopy(vehicle.impact)

        return state
        

    def load_vehicle_state(self, vehicle, saved_vehicle_state):
        vehicle.heading = copy.deepcopy(saved_vehicle_state["heading"])
        vehicle.position = copy.deepcopy(saved_vehicle_state["position"])
        vehicle.speed = copy.deepcopy(saved_vehicle_state["speed"])
        vehicle.target_speed = copy.deepcopy(saved_vehicle_state["target_speed"])
        vehicle.target_lane_index = copy.deepcopy(saved_vehicle_state["target_lane_index"])
        vehicle.speed_index = copy.deepcopy(saved_vehicle_state["speed_index"])
        vehicle.crashed = copy.deepcopy(saved_vehicle_state["crashed"])
        vehicle.collide_with = copy.deepcopy(saved_vehicle_state["collide_with"])
        vehicle.impact = copy.deepcopy(saved_vehicle_state["impact"])


    def step(self, vehicle, dt: float) -> None:
        """
        Propagate the vehicle state given its actions.

        Integrate a modified bicycle model with a 1st-order response on the steering wheel dynamics.
        If the vehicle is crashed, the actions are overridden with erratic steering and braking until complete stop.
        The vehicle's current lane is updated.

        :param dt: timestep of integration of the model [s]
        """
        vehicle.clip_actions()
        delta_f = vehicle.action['steering']
        # print("vehicle.action['acceleration']: ", vehicle.action['acceleration'])
        beta = np.arctan(1 / 2 * np.tan(delta_f))
        v = vehicle.speed * np.array([np.cos(vehicle.heading + beta),
                                   np.sin(vehicle.heading + beta)])
        # print("vehicle v: ", v)
        vehicle.position += v * dt
        # print("position: ", vehicle.position)
        if vehicle.impact is not None:
            vehicle.position += vehicle.impact
            vehicle.crashed = True
            vehicle.impact = None
        vehicle.heading += vehicle.speed * np.sin(beta) / (vehicle.LENGTH / 2) * dt
        vehicle.speed += vehicle.action['acceleration'] * dt
        # self.on_state_update()


    def victim_handle_collisions(self, other, dt: float = 0) -> None:
        """
        Check for collision with another vehicle.

        :param other: the other vehicle or object
        :param dt: timestep to check for future collisions (at constant velocity)
        """
        if not (other.collidable):
            return
        intersecting, will_intersect, transition = self.env.victim._is_colliding(other, dt)
        # print(str(intersecting), str(will_intersect))
        if will_intersect or intersecting:
            return True
        return False

class FollowingVictimVulnerable(PerfectVictim):

    def select_action(self, attacker_actions):
        if self.vulnerablility():
            return 0 # lane left
                    
        return super().select_action(attacker_actions)
    
    def vulnerablility(self):
        vehicle_ahead = vehicle_behind = vehicle_left = vehicle_right = False
        victim_l_index = self.env.victim.lane_index
        for i in range(len(self.env.controlled_vehicles)):
            other_l_index = self.env.controlled_vehicles[i].lane_index
            if other_l_index[2] == victim_l_index[2]:
                # attacker is the same lane as the victim
                victim_local_x = self.env.road.network.get_lane(victim_l_index).local_coordinates(self.env.victim.position)[0]
                attacker_local_x = self.env.road.network.get_lane(victim_l_index).local_coordinates(self.env.controlled_vehicles[i].position)[0]
                if attacker_local_x > victim_local_x and (attacker_local_x - victim_local_x) <= self.env.config["close_vehicle_threshold"]:
                    vehicle_ahead = True
                elif attacker_local_x < victim_local_x and (victim_local_x - attacker_local_x) <= self.env.config["close_vehicle_threshold"]:
                    vehicle_behind = True
            elif other_l_index[2] < victim_l_index[2] and self.env.road.network.get_lane(other_l_index).is_reachable_from(self.env.victim.position):
                # attacker is on the left lane of the victim
                victim_local_x = self.env.road.network.get_lane(other_l_index).local_coordinates(self.env.victim.position)[0]
                attacker_local_x = self.env.road.network.get_lane(other_l_index).local_coordinates(self.env.controlled_vehicles[i].position)[0]
                if abs(victim_local_x - attacker_local_x) <= self.env.config["diving_beside"]:
                    vehicle_left = True
            elif other_l_index[2] > victim_l_index[2] and self.env.road.network.get_lane(other_l_index).is_reachable_from(self.env.victim.position):
                # attacker is on the right lane of the victim
                victim_local_x = self.env.road.network.get_lane(other_l_index).local_coordinates(self.env.victim.position)[0]
                attacker_local_x = self.env.road.network.get_lane(other_l_index).local_coordinates(self.env.controlled_vehicles[i].position)[0]
                if abs(victim_local_x - attacker_local_x) <= self.env.config["diving_beside"]:
                    vehicle_right = True
        if vehicle_left and vehicle_right:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("victim action overrided")
            return True
        
class SurroundVictimVulnerable(PerfectVictim):

    def select_action(self, attacker_actions):
        if self.vulnerablility():
            return 0 # lane left
                    
        return super().select_action(attacker_actions)
    
    def vulnerablility(self):
        vehicle_ahead = vehicle_behind = vehicle_left = vehicle_right = False
        victim_l_index = self.env.victim.lane_index
        for i in range(len(self.env.controlled_vehicles)):
            other_l_index = self.env.controlled_vehicles[i].lane_index
            if other_l_index[2] == victim_l_index[2]:
                # attacker is the same lane as the victim
                victim_local_x = self.env.road.network.get_lane(victim_l_index).local_coordinates(self.env.victim.position)[0]
                attacker_local_x = self.env.road.network.get_lane(victim_l_index).local_coordinates(self.env.controlled_vehicles[i].position)[0]
                if attacker_local_x > victim_local_x and (attacker_local_x - victim_local_x) <= self.env.config["close_vehicle_threshold"]:
                    vehicle_ahead = True
                elif attacker_local_x < victim_local_x and (victim_local_x - attacker_local_x) <= self.env.config["close_vehicle_threshold"]:
                    vehicle_behind = True
            elif other_l_index[2] < victim_l_index[2] and self.env.road.network.get_lane(other_l_index).is_reachable_from(self.env.victim.position):
                # attacker is on the left lane of the victim
                victim_local_x = self.env.road.network.get_lane(other_l_index).local_coordinates(self.env.victim.position)[0]
                attacker_local_x = self.env.road.network.get_lane(other_l_index).local_coordinates(self.env.controlled_vehicles[i].position)[0]
                if abs(victim_local_x - attacker_local_x) <= self.env.config["diving_beside"]:
                    vehicle_left = True
            elif other_l_index[2] > victim_l_index[2] and self.env.road.network.get_lane(other_l_index).is_reachable_from(self.env.victim.position):
                # attacker is on the right lane of the victim
                victim_local_x = self.env.road.network.get_lane(other_l_index).local_coordinates(self.env.victim.position)[0]
                attacker_local_x = self.env.road.network.get_lane(other_l_index).local_coordinates(self.env.controlled_vehicles[i].position)[0]
                if abs(victim_local_x - attacker_local_x) <= self.env.config["diving_beside"]:
                    vehicle_right = True
        if vehicle_left and vehicle_right and vehicle_ahead and vehicle_behind:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("victim action overrided")
            return True
        

class FrontVictimVulnerable(PerfectVictim):

    def select_action(self, attacker_actions):
        if self.vulnerablility():
            return 0 # lane left
                    
        return super().select_action(attacker_actions)
    
    def vulnerablility(self):
        vehicle_ahead = vehicle_behind = vehicle_left = vehicle_right = False
        victim_l_index = self.env.victim.lane_index
        for i in range(len(self.env.controlled_vehicles)):
            other_l_index = self.env.controlled_vehicles[i].lane_index
            if other_l_index[2] == victim_l_index[2]:
                # attacker is the same lane as the victim
                victim_local_x = self.env.road.network.get_lane(victim_l_index).local_coordinates(self.env.victim.position)[0]
                attacker_local_x = self.env.road.network.get_lane(victim_l_index).local_coordinates(self.env.controlled_vehicles[i].position)[0]
                if attacker_local_x > victim_local_x and (attacker_local_x - victim_local_x) <= self.env.config["close_vehicle_threshold"]:
                    vehicle_ahead = True
                elif attacker_local_x < victim_local_x and (victim_local_x - attacker_local_x) <= self.env.config["close_vehicle_threshold"]:
                    vehicle_behind = True
            elif other_l_index[2] < victim_l_index[2] and self.env.road.network.get_lane(other_l_index).is_reachable_from(self.env.victim.position):
                # attacker is on the left lane of the victim
                victim_local_x = self.env.road.network.get_lane(other_l_index).local_coordinates(self.env.victim.position)[0]
                attacker_local_x = self.env.road.network.get_lane(other_l_index).local_coordinates(self.env.controlled_vehicles[i].position)[0]
                if abs(victim_local_x - attacker_local_x) <= self.env.config["diving_beside"]:
                    vehicle_left = True
            elif other_l_index[2] > victim_l_index[2] and self.env.road.network.get_lane(other_l_index).is_reachable_from(self.env.victim.position):
                # attacker is on the right lane of the victim
                victim_local_x = self.env.road.network.get_lane(other_l_index).local_coordinates(self.env.victim.position)[0]
                attacker_local_x = self.env.road.network.get_lane(other_l_index).local_coordinates(self.env.controlled_vehicles[i].position)[0]
                if abs(victim_local_x - attacker_local_x) <= self.env.config["diving_beside"]:
                    vehicle_right = True
        if vehicle_left and vehicle_right and vehicle_behind:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("victim action overrided")
            return True
        

class BackVictimVulnerable(PerfectVictim):

    def select_action(self, attacker_actions):
        if self.vulnerablility():
            return 0 # lane left
                    
        return super().select_action(attacker_actions)
    
    def vulnerablility(self):
        vehicle_ahead = vehicle_behind = vehicle_left = vehicle_right = False
        victim_l_index = self.env.victim.lane_index
        for i in range(len(self.env.controlled_vehicles)):
            other_l_index = self.env.controlled_vehicles[i].lane_index
            if other_l_index[2] == victim_l_index[2]:
                # attacker is the same lane as the victim
                victim_local_x = self.env.road.network.get_lane(victim_l_index).local_coordinates(self.env.victim.position)[0]
                attacker_local_x = self.env.road.network.get_lane(victim_l_index).local_coordinates(self.env.controlled_vehicles[i].position)[0]
                if attacker_local_x > victim_local_x and (attacker_local_x - victim_local_x) <= self.env.config["close_vehicle_threshold"]:
                    vehicle_ahead = True
                elif attacker_local_x < victim_local_x and (victim_local_x - attacker_local_x) <= self.env.config["close_vehicle_threshold"]:
                    vehicle_behind = True
            elif other_l_index[2] < victim_l_index[2] and self.env.road.network.get_lane(other_l_index).is_reachable_from(self.env.victim.position):
                # attacker is on the left lane of the victim
                victim_local_x = self.env.road.network.get_lane(other_l_index).local_coordinates(self.env.victim.position)[0]
                attacker_local_x = self.env.road.network.get_lane(other_l_index).local_coordinates(self.env.controlled_vehicles[i].position)[0]
                if abs(victim_local_x - attacker_local_x) <= self.env.config["diving_beside"]:
                    vehicle_left = True
            elif other_l_index[2] > victim_l_index[2] and self.env.road.network.get_lane(other_l_index).is_reachable_from(self.env.victim.position):
                # attacker is on the right lane of the victim
                victim_local_x = self.env.road.network.get_lane(other_l_index).local_coordinates(self.env.victim.position)[0]
                attacker_local_x = self.env.road.network.get_lane(other_l_index).local_coordinates(self.env.controlled_vehicles[i].position)[0]
                if abs(victim_local_x - attacker_local_x) <= self.env.config["diving_beside"]:
                    vehicle_right = True
        if vehicle_left and vehicle_right and vehicle_ahead:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("victim action overrided")
            return True

