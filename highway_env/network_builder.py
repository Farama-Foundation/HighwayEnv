import math
from enum import Enum

import numpy as np
from highway_env.utils import Vector
from highway_env.road.lanes.abstract_lanes import AbstractLane
from highway_env.road.lanes.lane_utils import LineType
from highway_env.road.lanes.unweighted_lanes import StraightLane, SineLane, CircularLane
from highway_env.road.road import RoadNetwork

class Path:
    def __init__(
        self,
        from_node_id: str,
        to_node_id: str,
        line_types: tuple[LineType, LineType],
        priority: int = 0,
        speed_limit: float = 20,
        forbidden: bool = False,
        width: float = AbstractLane.DEFAULT_WIDTH,
        weight: int = None
    ):
        """
        Parameters
        ----------
        from_node_id : str
            The id of the start point
        to_node_id : str
            The id of the end point
        line_types : tuple[LineType, LineType], optional
            The description of the line types in the road<br>
            (Default is ``None``)
        priority : int, optional
            Describing the priority of the road, higher value indicates higher priority<br>
            (Default is ``0``)
        speed_limit : float, optional
            Assign a speed limit in m/s<br>
            (Default is ``20`` (m/s))
        forbidden : bool, optional
            Assign if it is forbidden to move into the road by lane change<br>
            ``True`` -> it is forbidden to enter<br>
            ``False`` -> it is not forbidden to enter<br>
            (Default is ``False``)
        width : float, optional
            The width of a lane<br>
            (Default is ``AbstractLane.DEFAULT_WIDTH`` => ``4``)
        weight: int, optiona
            The weight of a lane<br>
            (Default is ``None``)
        """
        self.from_node_id: str = from_node_id
        self.to_node_id: str = to_node_id
        self.line_types: tuple[LineType, LineType] = line_types
        self.priority: int = priority
        self.speed_limit: float = speed_limit
        self.forbidden: bool = forbidden
        self.width: float = width,
        self.weight: int = weight

class StraightPath(Path):
    def __init__(
        self,
        from_node_id: str,
        to_node_id: str,
        line_types: tuple[LineType, LineType],
        weight: int = None,
        priority: int = 0,
        speed_limit: float = 20,
        forbidden: bool = False,
        width: float = AbstractLane.DEFAULT_WIDTH,
    ):
        """
        Parameters
        ----------
        from_node_id : str
            The id of the start point
        to_node_id : str
            The id of the end point
        line_types : tuple[LineType, LineType], optional
            The description of the line types in the road<br>
            (Default is ``None``)
        priority : int, optional
            Describing the priority of the road, higher value indicates higher priority<br>
            (Default is ``0``)
        speed_limit : float, optional
            Assign a speed limit in m/s<br>
            (Default is ``20`` (m/s))
        forbidden : bool, optional
            Assign if it is forbidden to move into the road by lane change<br>
            ``True`` -> it is forbidden to enter<br>
            ``False`` -> it is not forbidden to enter<br>
            (Default is ``False``)
        width : float, optional
            The width of a lane<br>
            (Default is ``AbstractLane.DEFAULT_WIDTH`` => ``4``)
        weight: int, optiona
            The weight of a lane<br>
            (Default is ``None``)
        """
        super().__init__(
            from_node_id,
            to_node_id,
            line_types,
            priority,
            speed_limit,
            forbidden,
            width,
            weight
        )
        
class CircularPath(Path):
    left_turn = False
    right_turn = True
    
    def __init__(
        self,
        from_node_id: str,
        to_node_id: str,
        start_phase: float, # degree
        end_phase: float, # degree
        line_types: tuple[LineType, LineType] = None,
        weight: int = None,
        priority: int = 0,
        speed_limit: float = 20,
        forbidden: bool = False,
        width: float = AbstractLane.DEFAULT_WIDTH,
    ):
        """
        Parameters
        ----------
        from_node_id : str
            The id of the start point
        to_node_id : str
            The id of the end point
        start_phase : float
            The starting phase<br>
            Note: ``0`` degrees is always upwards, the builder will handle when this is not the case
        end_phase : float
            The ending phase<br>
            Note: ``0`` degrees is always upwards, the builder will handle when this is not the case
        line_types : tuple[LineType, LineType], optional
            The description of the line types in the road<br>
            (Default is ``None``)
        clockwise : bool, optional
            Describing if the cirlce moves clockwise or counterclockwise<br>
            ``True`` -> clockwise<br>
            ``False`` -> counterclockwise<br>
            (Default is ``True``)
        priority : int, optional
            Describing the priority of the road, higher value indicates higher priority<br>
            (Default is ``0``)
        speed_limit : float, optional
            Assign a speed limit in m/s<br>
            (Default is ``20`` (m/s))
        forbidden : bool, optional
            Assign if it is forbidden to move into the road by lane change<br>
            ``True`` -> it is forbidden to enter<br>
            ``False`` -> it is not forbidden to enter<br>
            (Default is ``False``)
        width : float, optional
            The width of a lane<br>
            (Default is ``AbstractLane.DEFAULT_WIDTH`` => ``4``)
        weight: int, optiona
            The weight of a lane<br>
            (Default is ``None``)
        """
        super().__init__(
            from_node_id,
            to_node_id,
            line_types,
            priority,
            speed_limit,
            forbidden,
            width,
            weight
        )
        self.clockwise: bool = self._determine_turn_direction(start_phase, end_phase) # True->right_turn, False->left_turn
        
        self.start_phase = start_phase if not self.clockwise else start_phase + 180
        self.end_phase   = end_phase   if not self.clockwise else end_phase   + 180
            
        
    def _determine_turn_direction(self, start_phase: float, end_phase: float) -> bool:
        # Normalize the degrees to be between 0 and 360
        start_phase = start_phase % 360
        end_phase = end_phase % 360
        
        # Calculate the difference
        diff = (end_phase - start_phase) % 360
        
        if diff == 0:
            return None  # No movement
        return diff < 180
    
class SinePath(Path):
    def __init__(
        self,
        from_node_id: str,
        to_node_id: str,
        weight: int = None
    ):
        super().__init__(from_node_id, to_node_id, weight)

class NetworkBuilder:
    class PathType(Enum):
        STRAIGHT = "straight"
        CIRCULAR = "circular"
        SINE     = "sine"
    class PathPriority(Enum):
        NORTH_SOUTH = "ns"
        EAST_WEST   = "ew"
    class IntersectionDirection(Enum):
        NORTH = "north"
        SOUTH = "south"
        EAST  = "east"
        WEST  = "west"
    
    def __init__(
        self
    ):
        self._nodes: dict[str, Vector] = {}
        self._road_description: dict[NetworkBuilder.PathType, list[Path]] = {
            self.PathType.STRAIGHT : [],
            self.PathType.CIRCULAR : [],
            self.PathType.SINE     : []
        }

    # Perpendicular Bisector
    def _find_circle_center(self, start: Vector, end: Vector) -> Vector:
        x1, y1 = start
        x2, y2 = end
        
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        if x1 == x2:
            return (mid_x, None)
        elif y1 == y2:
            return (None, mid_y)

        slope = (y2 - y1) / (x2 - x1)
        perp_slope = -1 / slope

        radius = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2

        dx = math.sqrt(radius**2 / (1 + perp_slope**2))
        dy = perp_slope * dx

        center1 = (mid_x + dx, mid_y + dy)
        center2 = (mid_x - dx, mid_y - dy)

        return center1, center2

    def _cross_product(self, v1: Vector, v2: Vector) -> float:
        """Compute the 2D cross product of two vectors v1 and v2."""
        return v1[0] * v2[1] - v1[1] * v2[0]

    def _select_center(
        self,
        start: Vector,
        end: Vector,
        center1: Vector,
        center2: Vector,
        clockwise: bool
        ) -> Vector:
        
        start_x, start_y = start
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        vector_start_to_mid = (mid_x - start_x, mid_y - start_y)
        vector_mid_to_center1 = (center1[0] - mid_x, center1[1] - mid_y)
        
        cross = self._cross_product(vector_start_to_mid, vector_mid_to_center1)
        
        # Check direction based on the sign of the cross product
        if not clockwise:
            # For clockwise, cross product should be negative
            if cross < 0:
                return center1
            else:
                return center2
        else:
            # For counterclockwise, cross product should be positive
            if cross > 0:
                return center1
            else:
                return center2

    def _get_center(self, start: Vector, end: Vector, clockwise: bool) -> Vector:
        # clockwise :: False->left_turn ; True->right_turn
        center1, center2 = self._find_circle_center(start, end)
        
        return self._select_center(start, end, center1, center2, clockwise)

    def _get_radius(self, start: Vector, center: Vector) -> float:
            return math.sqrt((start[0] - center[0])**2 + (start[1] - center[1])**2)
    
    def add_node(self, id: str, coordinate: Vector):
        self._nodes.update({id: coordinate})
        
    def add_multiple_nodes(self, nodes: dict[str, Vector]):
        self._nodes.update(nodes)
        
    def add_road(self, path_type: PathType, road: Path):
        self._road_description[path_type].append(road)
        
    def add_multiple_roads(self, roads: dict[PathType, list[Path]]):
        for key in NetworkBuilder.PathType:
            if key in roads:
                self._road_description[key].extend(roads[key])
        
    # @TODO Determine the weight of roads inside the intersection
    def add_intersection(
        self,
        intersection_name: str,
        ingoing_roads: dict[IntersectionDirection, Vector],
        priority: PathPriority,
        lane_width: float = AbstractLane.DEFAULT_WIDTH
        ):
        """
        Assumes the I/O directions are North, South, East, and West.
        """

        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        nodes: dict[str, Vector] = {}
        
        for direction_enum, vector in ingoing_roads.items():
            direction = direction_enum.value[0].lower()
            for road_type in ['in', 'out']:
                node_name = f"{intersection_name}:{direction}-{road_type}"

                if road_type == "in":
                    value = vector
                else:  # road_type == "out"
                    x, y = vector
                    if direction_enum   == self.IntersectionDirection.NORTH:
                        x += lane_width
                    elif direction_enum == self.IntersectionDirection.SOUTH:
                        x -= lane_width
                    elif direction_enum == self.IntersectionDirection.EAST:
                        y += lane_width
                    elif direction_enum == self.IntersectionDirection.WEST:
                        y -= lane_width
                    value = [x, y]
                nodes[node_name] = value
            
        self.add_multiple_nodes(nodes)
        
        
        
        # Initialize road_desc
        road_desc = {
            self.PathType.STRAIGHT: [],
            self.PathType.CIRCULAR: []
        }

        # Define opposite directions
        opposite_directions = {
            self.IntersectionDirection.NORTH: self.IntersectionDirection.SOUTH,
            self.IntersectionDirection.SOUTH: self.IntersectionDirection.NORTH,
            self.IntersectionDirection.EAST:  self.IntersectionDirection.WEST,
            self.IntersectionDirection.WEST:  self.IntersectionDirection.EAST
        }

        # Define priority pairs based on the priority parameter
        if priority == self.PathPriority.NORTH_SOUTH:
            priority_pairs = {
                (self.IntersectionDirection.NORTH, self.IntersectionDirection.SOUTH),
                (self.IntersectionDirection.SOUTH, self.IntersectionDirection.NORTH)
            }
        elif priority == self.PathPriority.EAST_WEST:
            priority_pairs = {
                (self.IntersectionDirection.EAST, self.IntersectionDirection.WEST),
                (self.IntersectionDirection.WEST, self.IntersectionDirection.EAST)
            }
        else:
            priority_pairs = set()

        # Mapping from directions to phases (degrees)
        direction_to_phase = {
            self.IntersectionDirection.NORTH: 180,
            self.IntersectionDirection.EAST : 270,
            self.IntersectionDirection.SOUTH: 0,
            self.IntersectionDirection.WEST : 90
        }

        # Define right and left turns
        left_turns = {
            (self.IntersectionDirection.NORTH, self.IntersectionDirection.EAST),
            (self.IntersectionDirection.EAST, self.IntersectionDirection.SOUTH),
            (self.IntersectionDirection.SOUTH, self.IntersectionDirection.WEST),
            (self.IntersectionDirection.WEST, self.IntersectionDirection.NORTH)
        }

        right_turns = {
            (self.IntersectionDirection.NORTH, self.IntersectionDirection.WEST),
            (self.IntersectionDirection.WEST, self.IntersectionDirection.SOUTH),
            (self.IntersectionDirection.SOUTH, self.IntersectionDirection.EAST),
            (self.IntersectionDirection.EAST, self.IntersectionDirection.NORTH)
        }

        
        directions = list(ingoing_roads.keys())
        for from_direction in directions:
            from_direction_char = from_direction.value[0].lower()
            from_node_in = f"{intersection_name}:{from_direction_char}-in"
            for to_direction in directions:
                if to_direction == from_direction:
                    continue  # Skip paths to the same direction
                
                to_direction_char = to_direction.value[0].lower()
                to_node_out = f"{intersection_name}:{to_direction_char}-out"

                if to_direction == opposite_directions[from_direction]:
                    # StraightPath
                    if (from_direction, to_direction) in priority_pairs:
                        lane_priority = 3
                        
                        # Determine if the inner line should be a striped or none
                        if priority == self.PathPriority.NORTH_SOUTH and from_direction == self.IntersectionDirection.NORTH\
                            or priority == self.PathPriority.EAST_WEST and from_direction == self.IntersectionDirection.EAST:
                            
                            first_value = s
                        else:
                            first_value = n  # Default case
                            
                            
                        # Determine if the outer line should be continuous or none depending on if there is a right turn
                        has_right_turn = False
                        
                        for to_dir in directions:
                            if to_dir == from_direction:
                                continue
                            
                            if (from_direction, to_dir) in right_turns:
                                has_right_turn = True
                                break
                        
                        second_value = n if has_right_turn else c

                        line_types = (first_value, second_value)
                        
                    else:
                        line_types = (n, n)
                        lane_priority = 0
                        
                    path = StraightPath(
                        from_node_in,
                        to_node_out,
                        line_types,
                        lane_priority,
                        lane_width
                    )
                    road_desc[self.PathType.STRAIGHT].append(path)
                else:
                    # CircularPath
                    line_types = (c, c)
                    
                    # Determine the lane priority
                    if (from_direction, to_direction) in right_turns:
                        lane_priority = 2
                    else:
                        lane_priority = 1
                    
                    # Determine start_phase and end_phase
                    start_phase = direction_to_phase[from_direction]
                    if (from_direction, to_direction) in right_turns:
                        end_phase = start_phase + 90
                        line_types = (n,c)
                    elif (from_direction, to_direction) in left_turns:
                        end_phase = start_phase - 90
                        line_types = (n,n)
                    else:
                        continue  # Skip invalid direction combinations
                    

                    path = CircularPath(
                        from_node_in,
                        to_node_out,
                        start_phase,
                        end_phase,
                        line_types,
                        lane_priority,
                        width=lane_width
                    )
                    road_desc[self.PathType.CIRCULAR].append(path)
                    print()
        
        # Add the constructed roads to the network
        self.add_multiple_roads(road_desc)
        
    def add_roundabout(self):
        print("Not implemented")
        
    def _build_straight_road(self, road: StraightPath) -> tuple[str, str, AbstractLane, int]:
        return (
            road.from_node_id,
            road.to_node_id,
            StraightLane(
                self._nodes[road.from_node_id],
                self._nodes[road.to_node_id],
                road.width,
                road.line_types,
                road.forbidden,
                road.speed_limit,
                road.priority,
            ),
            road.weight
        )
        
    def _build_circular_road(self, road: CircularPath) -> tuple[str, str, AbstractLane, int]:
        center = self._get_center(
            self._nodes[road.from_node_id],
            self._nodes[road.to_node_id],
            road.clockwise
        )
        
        return (
            road.from_node_id,
            road.to_node_id,
            CircularLane(
                center,
                self._get_radius(self._nodes[road.from_node_id], center),
                np.deg2rad(road.start_phase),
                np.deg2rad(road.end_phase),
                road.clockwise,
                road.width,
                road.line_types,
                road.forbidden,
                road.speed_limit,
                road.priority
            ),
            road.weight
        )
    
    def _build_sine_road(self, road: SinePath) -> tuple[str, str, AbstractLane, int]:
        return ()
    
    def build_roads(self, road_network: RoadNetwork):
        net: list[tuple[str, str, AbstractLane]] = []

        # Mapping road types to their respective build methods
        build_methods = {
            self.PathType.STRAIGHT: self._build_straight_road,
            self.PathType.CIRCULAR: self._build_circular_road,
            self.PathType.SINE: self._build_sine_road
        }

        # Iterate through each road type and description
        for road_type, build_method in build_methods.items():
            for road in self._road_description[road_type]:
                net.append(build_method(road))

        # Add each lane to the road network
        for from_id, to_id, lane, weight in net:
            road_network.add_lane(from_id, to_id, lane, weight)
