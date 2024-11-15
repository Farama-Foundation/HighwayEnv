import math
import re
from enum import Enum

import numpy as np
from highway_env.utils import Vector
from highway_env.road.lanes.abstract_lanes import AbstractLane
from highway_env.road.lanes.lane_utils import LaneType, LineType
from highway_env.road.lanes.unweighted_lanes import StraightLane, SineLane, CircularLane
from highway_env.road.road import RoadNetwork

class Path:
    def __init__(
        self,
        from_node_id: str,
        to_node_id: str,
        line_types: tuple[LineType, LineType],
        priority: int = 0,
        speed_limit: float = 40,
        forbidden: bool = False,
        width: float = AbstractLane.DEFAULT_WIDTH,
        weight: int = None,
        lane_type: LaneType = None,
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
            (Default is ``40`` (m/s))
        forbidden : bool, optional
            Assign if it is forbidden to move into the road by lane change<br>
            ``True`` -> it is forbidden to enter<br>
            ``False`` -> it is not forbidden to enter<br>
            (Default is ``False``)
        width : float, optional
            The width of a lane<br>
            (Default is ``AbstractLane.DEFAULT_WIDTH`` => ``4``)
        weight: int, optional
            The weight of a lane<br>
            (Default is ``None``)
        lane_type: LaneType, optional
            Describing the type of the lane, e.g. a highway, roundabout, or intersection.<br>
            This can be used when training a model to give it the knowledge of where it is<br>
            driving which can be used to alter its behaviour, e.g. how fast it drives.<br>
            (Default is ``None``) 
        """
        self.from_node_id: str = from_node_id
        self.to_node_id: str = to_node_id
        self.line_types: tuple[LineType, LineType] = line_types
        self.priority: int = priority
        self.speed_limit: float = speed_limit
        self.forbidden: bool = forbidden
        self.width: float = width
        self.weight: int = weight
        self.lane_type: LaneType = lane_type

class StraightPath(Path):
    def __init__(
        self,
        from_node_id: str,
        to_node_id: str,
        line_types: tuple[LineType, LineType],
        weight: int = None,
        lane_type: LaneType = None,
        priority: int = 0,
        speed_limit: float = 40,
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
        weight: int, optional
            The weight of a lane<br>
            (Default is ``None``)
        lane_type: LaneType, optional
            Describing the type of the lane, e.g. a highway, roundabout, or intersection.<br>
            This can be used when training a model to give it the knowledge of where it is<br>
            driving which can be used to alter its behaviour, e.g. how fast it drives.<br>
            (Default is ``None``)            
        priority : int, optional
            Describing the priority of the road, higher value indicates higher priority<br>
            (Default is ``0``)
        speed_limit : float, optional
            Assign a speed limit in m/s<br>
            (Default is ``40`` (m/s))
        forbidden : bool, optional
            Assign if it is forbidden to move into the road by lane change<br>
            ``True`` -> it is forbidden to enter<br>
            ``False`` -> it is not forbidden to enter<br>
            (Default is ``False``)
        width : float, optional
            The width of a lane<br>
            (Default is ``AbstractLane.DEFAULT_WIDTH`` => ``4``)
        """
        super().__init__(
            from_node_id,
            to_node_id,
            line_types,
            priority,
            speed_limit,
            forbidden,
            width,
            weight,
            lane_type
        )

class CircularPath(Path):
    left_turn = False
    right_turn = True
    
    def __init__(
        self,
        from_node_id: str,
        to_node_id: str,
        start_phase: float, # degree
        radius: float,
        clockwise: bool, # True->right_turn, False->left_turn
        line_types: tuple[LineType, LineType] = None,
        weight: int = None,
        lane_type: LaneType = None,
        priority: int = 0,
        speed_limit: float = 40,
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
        radius : float
            The ending phase<br>
            Note: ``0`` degrees is always upwards, the builder will handle when this is not the case
        clockwise : bool
            Describing if the cirlce moves clockwise or counterclockwise<br>
            ``True`` -> clockwise<br>
            ``False`` -> counterclockwise<br>
        line_types : tuple[LineType, LineType], optional
            The description of the line types in the road<br>
            (Default is ``None``)
        weight : int, optional
            The weight of a lane<br>
            (Default is ``None``)
        lane_type : LaneType, optional
            Describing the type of the lane, e.g. a highway, roundabout, or intersection.<br>
            This can be used when training a model to give it the knowledge of where it is<br>
            driving which can be used to alter its behaviour, e.g. how fast it drives.<br>
            (Default is ``None``) 
        priority : int, optional
            Describing the priority of the road, higher value indicates higher priority<br>
            (Default is ``0``)
        speed_limit : float, optional
            Assign a speed limit in m/s<br>
            (Default is ``40`` (m/s))
        forbidden : bool, optional
            Assign if it is forbidden to move into the road by lane change<br>
            ``True`` -> it is forbidden to enter<br>
            ``False`` -> it is not forbidden to enter<br>
            (Default is ``False``)
        width : float, optional
            The width of a lane<br>
            (Default is ``AbstractLane.DEFAULT_WIDTH`` => ``4``)

        """
        super().__init__(
            from_node_id,
            to_node_id,
            line_types,
            priority,
            speed_limit,
            forbidden,
            width,
            weight,
            lane_type
        )
        
        self.clockwise: bool    = clockwise
        self.radius: float      = radius
        self.start_phase: float = start_phase if not self.clockwise else start_phase + 180
        
        
    
    def _get_distance(self, point_a: Vector, point_b: Vector) -> float:
        x1, y1 = point_a
        x2, y2 = point_b
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def get_phase(self, point_a: Vector, point_b: Vector, radius: float, clockwise: bool) -> float:
        d: float = self._get_distance(point_a, point_b)
        theta: float = 2 * math.asin(d / (2 * radius))
        phase: float = theta * 180 / np.pi

        return phase if not clockwise else phase + 180
    
    def get_center(self, point_a: Vector, point_b: Vector, len_a: float, len_b: float, clockwise: bool) -> Vector:
        x1, y1 = point_a
        x2, y2 = point_b
        
        len_c: float = self._get_distance(point_a, point_b)
        
        unit_vector: Vector = [(x2 - x1) / len_c, (y2 - y1) / len_c]
        perp_vector: Vector = [-((y2 - y1) / len_c), (x2 - x1) / len_c]
        
        
        cos_theta: float = (len_b**2 + len_c**2 - len_a**2) / (2 * len_b * len_c)
        cos_theta_sq = cos_theta**2

        sin_theta = math.sqrt(1 - cos_theta_sq)
        
        x3_plus  = round(x1 + len_b * cos_theta * unit_vector[0] + len_b * sin_theta * perp_vector[0], 4) + 0.0 # to handle -0.0 cases
        x3_minus = round(x1 + len_b * cos_theta * unit_vector[0] - len_b * sin_theta * perp_vector[0], 4) + 0.0 # to handle -0.0 cases
        y3_plus  = round(y1 + len_b * cos_theta * unit_vector[1] + len_b * sin_theta * perp_vector[1], 4) + 0.0 # to handle -0.0 cases
        y3_minus = round(y1 + len_b * cos_theta * unit_vector[1] - len_b * sin_theta * perp_vector[1], 4) + 0.0 # to handle -0.0 cases
        
        return [x3_plus, y3_plus] if clockwise else [x3_minus, y3_minus]
    
class SinePath(Path):
    def __init__(
        self,
        from_node_id: str,
        to_node_id: str,
        line_types: tuple[LineType, LineType],
        priority: int = 0,
        speed_limit: float = 40,
        forbidden: bool = False,
        width: float = AbstractLane.DEFAULT_WIDTH,
        weight: int = None,
        lane_type: LaneType = None
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
        weight: int, optional
            The weight of a lane<br>
            (Default is ``None``)
        lane_type: LaneType, optional
            Describing the type of the lane, e.g. a highway, roundabout, or intersection.<br>
            This can be used when training a model to give it the knowledge of where it is<br>
            driving which can be used to alter its behaviour, e.g. how fast it drives.<br>
            (Default is ``None``) 
        priority : int, optional
            Describing the priority of the road, higher value indicates higher priority<br>
            (Default is ``0``)
        speed_limit : float, optional
            Assign a speed limit in m/s<br>
            (Default is ``40`` (m/s))
        forbidden : bool, optional
            Assign if it is forbidden to move into the road by lane change<br>
            ``True`` -> it is forbidden to enter<br>
            ``False`` -> it is not forbidden to enter<br>
            (Default is ``False``)
        width : float, optional
            The width of a lane<br>
            (Default is ``AbstractLane.DEFAULT_WIDTH`` => ``4``)
        """
        super().__init__(
            from_node_id,
            to_node_id,
            line_types,
            priority,
            speed_limit,
            forbidden,
            width,
            weight,
            lane_type)
        

class NetworkBuilder:
    class PathType(Enum):
        STRAIGHT = "straight"
        CIRCULAR = "circular"
        SINE     = "sine"
    class PathPriority(Enum):
        NORTH_SOUTH = "ns"
        EAST_WEST   = "ew"
    class CardinalDirection(Enum):
        NORTH = "north"
        SOUTH = "south"
        EAST  = "east"
        WEST  = "west"
    
    def __init__(
        self
    ):
        """
        Naming convention
        -----------------
            This framework, and subsequently this method, use the naming convention<br>
            ``<type>-<id<:<direction>-<in|out>[:lane_index]``. This method will thus generate the ``in|out`` nodes for each direction.
            <br><br>
            - ``type`` refers to the type of the road, e.g. "``I``" for "intersection" or "``T``" for "turn".<br>
            - ``id`` refers to the id of the type, e.g. the first intersection is ``I-1`` and the second is ``I-2``.<br>
            - ``direction`` refers to the cardinal traversal direction of the road which are about to be made.<br>
            - ``in|out`` refers to the node either being an ``in`` node or ``out`` node. An ``In`` node is the<br>
                node where you enter the road and the ``out`` node is the node where you exit the road.<br>
                E.g. ``in`` nodes in an intersection are the ones which enters the intersection<br>
                while the ``out`` nodes are the ones which exits the intersection.<br>
            - ``:lane_index`` refers to the lane's index such that it is possible<br>
                to have multiple lanes from between the same nodes.<br>
                The ``:``is a separator between the direction part and the lane indexation<br>
                The ``:lane_index`` will be removed from the node name when building<br>
                such that the network will know that the lanes are connected and<br>
                it is possible to move between them. However, the NetworkBuilder will<br>
                still store in its dictionary of nodes and paths the ``:lane_index``<br>
                you provide - thus it is <b><i>only</i></b> when building the network it removes the lane index
        """
        self._nodes: dict[str, Vector] = {}
        self._path_description: dict[NetworkBuilder.PathType, list[Path]] = {
            self.PathType.STRAIGHT : [],
            self.PathType.CIRCULAR : [],
            self.PathType.SINE     : []
        }
    
    def add_node(self, id: str, coordinate: Vector):
        """
        Naming convention
        -----------------
            This framework, and subsequently this method, use the naming convention<br>
            ``<type>-<id<:<direction>-<in|out>[:lane_index]``. This method will thus generate the ``in|out`` nodes for each direction.
            <br><br>
            - ``type`` refers to the type of the road, e.g. "``I``" for "intersection" or "``T``" for "turn".<br>
            - ``id`` refers to the id of the type, e.g. the first intersection is ``I-1`` and the second is ``I-2``.<br>
            - ``direction`` refers to the cardinal traversal direction of the road which are about to be made.<br>
            - ``in|out`` refers to the node either being an ``in`` node or ``out`` node. An ``In`` node is the<br>
                node where you enter the road and the ``out`` node is the node where you exit the road.<br>
                E.g. ``in`` nodes in an intersection are the ones which enters the intersection<br>
                while the ``out`` nodes are the ones which exits the intersection.<br>
            - ``:lane_index`` refers to the lane's index such that it is possible<br>
                to have multiple lanes from between the same nodes.<br>
                The ``:``is a separator between the direction part and the lane indexation<br>
                The ``:lane_index`` will be removed from the node name when building<br>
                such that the network will know that the lanes are connected and<br>
                it is possible to move between them. However, the NetworkBuilder will<br>
                still store in its dictionary of nodes and paths the ``:lane_index``<br>
                you provide - thus it is <b><i>only</i></b> when building the network it removes the lane index
        
        Description
        -----------
            Adds a single node to the NetworkBuilder's dictionary<br>
            which will be used when building the network using<br>
            ``self-build_paths(...)``
        
        Parameters
        ----------
        id : str
            The id of a node
        coordinate : Vector
            The location of the node
        
        Example
        -------
        ```python
        nb = NetworkBuilder()
        nb.add_node("road_start", [0, 0])
        nb.add_node("road_end", [200, 0])
        ```
        """
        self._nodes.update({id: coordinate})
        
    def add_multiple_nodes(self, nodes: dict[str, Vector]):
        """
        Naming convention
        -----------------
            This framework, and subsequently this method, use the naming convention<br>
            ``<type>-<id<:<direction>-<in|out>[:lane_index]``. This method will thus generate the ``in|out`` nodes for each direction.
            <br><br>
            - ``type`` refers to the type of the road, e.g. "``I``" for "intersection" or "``T``" for "turn".<br>
            - ``id`` refers to the id of the type, e.g. the first intersection is ``I-1`` and the second is ``I-2``.<br>
            - ``direction`` refers to the cardinal traversal direction of the road which are about to be made.<br>
            - ``in|out`` refers to the node either being an ``in`` node or ``out`` node. An ``In`` node is the<br>
                node where you enter the road and the ``out`` node is the node where you exit the road.<br>
                E.g. ``in`` nodes in an intersection are the ones which enters the intersection<br>
                while the ``out`` nodes are the ones which exits the intersection.<br>
            - ``:lane_index`` refers to the lane's index such that it is possible<br>
                to have multiple lanes from between the same nodes.<br>
                The ``:``is a separator between the direction part and the lane indexation<br>
                The ``:lane_index`` will be removed from the node name when building<br>
                such that the network will know that the lanes are connected and<br>
                it is possible to move between them. However, the NetworkBuilder will<br>
                still store in its dictionary of nodes and paths the ``:lane_index``<br>
                you provide - thus it is <b><i>only</i></b> when building the network it removes the lane index
        
        Description
        -----------
            Adds multiple nodes to the NetworkBuilder's dictionary<br>
            which will be used when building the network using<br>
            ``self-build_paths(...)``
            
        Parameters
        ----------
        nodes : dict[str, Vector]
            A dictionary containing the node's ``id: str`` and ``coordinate: Vector`` location
            
        Example
        -------
        ```python
        nb = NetworkBuilder()
        nb.add_multiple_nodes({
            "road_start": [0, 0],
            "road_end"  : [200, 0]
        })
        ```
        """
        self._nodes.update(nodes)
        
    def add_path(self, path_type: PathType, path: Path):
        """
        Naming convention
        -----------------
            This framework, and subsequently this method, use the naming convention<br>
            ``<type>-<id<:<direction>-<in|out>[:lane_index]``. This method will thus generate the ``in|out`` nodes for each direction.
            <br><br>
            - ``type`` refers to the type of the road, e.g. "``I``" for "intersection" or "``T``" for "turn".<br>
            - ``id`` refers to the id of the type, e.g. the first intersection is ``I-1`` and the second is ``I-2``.<br>
            - ``direction`` refers to the cardinal traversal direction of the road which are about to be made.<br>
            - ``in|out`` refers to the node either being an ``in`` node or ``out`` node. An ``In`` node is the<br>
                node where you enter the road and the ``out`` node is the node where you exit the road.<br>
                E.g. ``in`` nodes in an intersection are the ones which enters the intersection<br>
                while the ``out`` nodes are the ones which exits the intersection.<br>
            - ``:lane_index`` refers to the lane's index such that it is possible<br>
                to have multiple lanes from between the same nodes.<br>
                The ``:``is a separator between the direction part and the lane indexation<br>
                The ``:lane_index`` will be removed from the node name when building<br>
                such that the network will know that the lanes are connected and<br>
                it is possible to move between them. However, the NetworkBuilder will<br>
                still store in its dictionary of nodes and paths the ``:lane_index``<br>
                you provide - thus it is <b><i>only</i></b> when building the network it removes the lane index
        
        Description
        -----------
            Adds a single path to the NetworkBuilder's dictionary<br>
            which will be used when building the network using<br>
            ``self-build_paths(...)``

        Parameters
        ----------
        path_type : PathType
            An enum type describing what type of path is provided<br>
            which will be used as the key in the dictionary containing paths.<br>
            PathType contains 3 values; ``STRAIGHT``, ``CIRCULAR``, and ``SINE``.
        path : Path
            The path which gives the NetworkBuilder the necessary information to<br>
            be able to construct the network when building the network.

        Example
        -------
        ```python
        c = LineType.NONE, LineType.CONTINUOUS
        nb = NetworkBuilder()
        nb.add_path(
            PathType.STRAIGHT,
            StraightPath(
                "road_start",
                "road_end",
                (c,c),
                10, # weight
                LaneType.HIGHWAY # lane_type
        ))
        
        # Alternative way (one-line)
        nb.add_path(
            PathType.STRAIGHT,
            StraightPath("road_start", "road_end", (c,c), 10, LaneType.HIGHWAY)
        )
        ```
        """
        self._path_description[path_type].append(path)
        
    def add_multiple_paths(self, paths: dict[PathType, list[Path]]):
        """
        Naming convention
        -----------------
            This framework, and subsequently this method, use the naming convention<br>
            ``<type>-<id<:<direction>-<in|out>[:lane_index]``. This method will thus generate the ``in|out`` nodes for each direction.
            <br><br>
            - ``type`` refers to the type of the road, e.g. "``I``" for "intersection" or "``T``" for "turn".<br>
            - ``id`` refers to the id of the type, e.g. the first intersection is ``I-1`` and the second is ``I-2``.<br>
            - ``direction`` refers to the cardinal traversal direction of the road which are about to be made.<br>
            - ``in|out`` refers to the node either being an ``in`` node or ``out`` node. An ``In`` node is the<br>
                node where you enter the road and the ``out`` node is the node where you exit the road.<br>
                E.g. ``in`` nodes in an intersection are the ones which enters the intersection<br>
                while the ``out`` nodes are the ones which exits the intersection.<br>
            - ``:lane_index`` refers to the lane's index such that it is possible<br>
                to have multiple lanes from between the same nodes.<br>
                The ``:``is a separator between the direction part and the lane indexation<br>
                The ``:lane_index`` will be removed from the node name when building<br>
                such that the network will know that the lanes are connected and<br>
                it is possible to move between them. However, the NetworkBuilder will<br>
                still store in its dictionary of nodes and paths the ``:lane_index``<br>
                you provide - thus it is <b><i>only</i></b> when building the network it removes the lane index
                
        Description
        -----------
            Adds multiple paths to the NetworkBuilder's dictionary<br>
            which will be used when building the network using<br>
            ``self.build_paths(...)``

        Parameters
        ----------
        paths : dict[PathType, list[Path]]
            A dictionary containing the type of the path (``PathType``) and<br>
            a list of paths describing how the nodes are connected.
        
        Example
        -------
        ```python
        c = LineType.NONE, LineType.CONTINUOUS
        nb = NetworkBuilder()
        nb.add_multiple_paths({
            PathType.STRAIGHT : [
                StraightPath(
                    "road_start",
                    "road_end",
                    (c,c),
                    10, # weight
                    LaneType.HIGHWAY # lane_type
                )
            ]
        })
        
        # Alternative way (one-line)
        nb.add_multiple_paths({
            PathType.STRAIGHT : [
                StraightPath("road_start", "road_end", (c,c), 10, LaneType.HIGHWAY)
            ]
        })
        ```
        """
        for key in NetworkBuilder.PathType:
            if key in paths:
                self._path_description[key].extend(paths[key])
        
    # @TODO Determine the weight of paths inside the intersection
    def add_intersection(
        self,
        intersection_name: str,
        ingoing_roads: dict[CardinalDirection, Vector],
        priority: PathPriority,
        weight: int = None,
        lane_width: float = AbstractLane.DEFAULT_WIDTH
        ):
        """
        Naming convention
        -----------------
            This framework, and subsequently this method, use the naming convention<br>
            ``<type>-<id<:<direction>-<in|out>[:lane_index]``. This method will thus generate the ``in|out`` nodes for each direction.
            <br><br>
            - ``type`` refers to the type of the road, e.g. "``I``" for "intersection" or "``T``" for "turn".<br>
            - ``id`` refers to the id of the type, e.g. the first intersection is ``I-1`` and the second is ``I-2``.<br>
            - ``direction`` refers to the cardinal traversal direction of the road which are about to be made.<br>
            - ``in|out`` refers to the node either being an ``in`` node or ``out`` node. An ``In`` node is the<br>
                node where you enter the road and the ``out`` node is the node where you exit the road.<br>
                E.g. ``in`` nodes in an intersection are the ones which enters the intersection<br>
                while the ``out`` nodes are the ones which exits the intersection.<br>
            - ``:lane_index`` refers to the lane's index such that it is possible<br>
                to have multiple lanes from between the same nodes.<br>
                The ``:``is a separator between the direction part and the lane indexation<br>
                The ``:lane_index`` will be removed from the node name when building<br>
                such that the network will know that the lanes are connected and<br>
                it is possible to move between them. However, the NetworkBuilder will<br>
                still store in its dictionary of nodes and paths the ``:lane_index``<br>
                you provide - thus it is <b><i>only</i></b> when building the network it removes the lane index
            
        Description
        -----------
            Adds an intersection to the road network, generating and configuring<br>
            incoming and outgoing lanes based on the specified direction,<br>
            priority, and lane width. This method establishes both straight and<br>
            circular paths between directions.

        Parameters
        ----------
            intersection_name : str
                A unique identifier for the intersection.
                
            ingoing_roads : dict[CardinalDirection, Vector]
                Maps directions (North, South, East, West) to their respective<br>
                vectors, which define the positions of ingoing roads.
            
            priority : PathPriority
                The prioritized lane direction (e.g., North-South or East-West).
            
            weight: int, optional
                The weight of a lane<br>
                (Default is ``None``)

            lane_width : float
                Width of each lane, with a default value from AbstractLane.

        Mapping of an ingoing point to other ingoing points
        ----------------------
        This asumes that you <b>do not</b> modify the lane width.<br>
        
        <b>North-in</b> -> <b>south-in</b> mapping: ``[+4, +12]`` <br>
        <b>North-in</b> -> <b>West-in</b> mapping: ``[-4, +8]`` <br>
        <b>North-in</b> -> <b>East-in</b> mapping: ``[+8, +4]`` <br><br>

        <b>South-in</b> -> <b>North-in</b> mapping: ``[-4, -12]`` <br>
        <b>South-in</b> -> <b>West-in</b> mapping: ``[-8, -4]`` <br>
        <b>South-in</b> -> <b>East-in</b> mapping: ``[+4, -8]`` <br><br>
        
        <b>West-in</b> -> <b>East-in</b> mapping: ``[+12, -4]`` <br>
        <b>West-in</b> -> <b>North-in</b> mapping: ``[+4, -8]`` <br>
        <b>West-in</b> -> <b>south-in</b> mapping: ``[+8, +4]`` <br><br>
        
        <b>East-in</b> -> <b>West-in</b> mapping: ``[-12, +4]`` <br>
        <b>East-in</b> -> <b>North-in</b> mapping: ``[-8, -4]`` <br>
        <b>East-in</b> -> <b>south-in</b> mapping: ``[-4 , +8]`` <br>

        Example
        -------
        ```python
        nb = NetworkBuilder()
        nb.add_intersection(
            "I-2",
            {
                net_builder.CardinalDirection.NORTH : [64, -92],
                net_builder.CardinalDirection.SOUTH : [68, -80],
                net_builder.CardinalDirection.WEST  : [60, -84],
            },
            net_builder.PathPriority.NORTH_SOUTH
        )
        # This will generate the following nodes:
        # "I-2:n-in"
        # "I-2:s-in"
        # "I-2:w-in"
        # "I-2:n-out"
        # "I-2:s-out"
        # "I-2:w-out"
        ```
        
        Assumptions
        -----------
            The directions are expected to be the cardinal directions North, South,<br>
            East, and West. This method manages the road layout to handle<br>
            straight and turning paths based on priorities and intersection geometry.
        """

        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        left_turn = False
        right_turn = True
        nodes: dict[str, Vector] = {}
        
        for direction_enum, vector in ingoing_roads.items():
            direction = direction_enum.value[0].lower()
            for road_type in ['in', 'out']:
                node_name = f"{intersection_name}:{direction}-{road_type}"

                if road_type == "in":
                    value = vector
                else:  # road_type == "out"
                    x, y = vector
                    if direction_enum   == self.CardinalDirection.NORTH:
                        x += lane_width
                    elif direction_enum == self.CardinalDirection.SOUTH:
                        x -= lane_width
                    elif direction_enum == self.CardinalDirection.EAST:
                        y += lane_width
                    elif direction_enum == self.CardinalDirection.WEST:
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
            self.CardinalDirection.NORTH: self.CardinalDirection.SOUTH,
            self.CardinalDirection.SOUTH: self.CardinalDirection.NORTH,
            self.CardinalDirection.EAST:  self.CardinalDirection.WEST,
            self.CardinalDirection.WEST:  self.CardinalDirection.EAST
        }

        # Define priority pairs based on the priority parameter
        if priority == self.PathPriority.NORTH_SOUTH:
            priority_pairs = {
                (self.CardinalDirection.NORTH, self.CardinalDirection.SOUTH),
                (self.CardinalDirection.SOUTH, self.CardinalDirection.NORTH)
            }
        elif priority == self.PathPriority.EAST_WEST:
            priority_pairs = {
                (self.CardinalDirection.EAST, self.CardinalDirection.WEST),
                (self.CardinalDirection.WEST, self.CardinalDirection.EAST)
            }
        else:
            priority_pairs = set()

        # Mapping from directions to phases (degrees)
        direction_to_phase = {
            self.CardinalDirection.NORTH: 180,
            self.CardinalDirection.EAST : 270,
            self.CardinalDirection.SOUTH: 0,
            self.CardinalDirection.WEST : 90
        }

        # Define right and left turns
        left_turns = {
            (self.CardinalDirection.NORTH, self.CardinalDirection.EAST),
            (self.CardinalDirection.EAST, self.CardinalDirection.SOUTH),
            (self.CardinalDirection.SOUTH, self.CardinalDirection.WEST),
            (self.CardinalDirection.WEST, self.CardinalDirection.NORTH)
        }

        right_turns = {
            (self.CardinalDirection.NORTH, self.CardinalDirection.WEST),
            (self.CardinalDirection.WEST, self.CardinalDirection.SOUTH),
            (self.CardinalDirection.SOUTH, self.CardinalDirection.EAST),
            (self.CardinalDirection.EAST, self.CardinalDirection.NORTH)
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
                        if priority == self.PathPriority.NORTH_SOUTH and from_direction == self.CardinalDirection.NORTH\
                            or priority == self.PathPriority.EAST_WEST and from_direction == self.CardinalDirection.EAST:
                            
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
                        line_types = (n,c)
                        
                        path = CircularPath(
                            from_node_in,
                            to_node_out,
                            start_phase,
                            lane_width,
                            right_turn,
                            line_types,
                            weight,
                            LaneType.INTERSECTION,
                            lane_priority,
                            width=lane_width
                        )
                        
                    elif (from_direction, to_direction) in left_turns:
                        line_types = (n,n)

                        path = CircularPath(
                            from_node_in,
                            to_node_out,
                            start_phase,
                            2* lane_width,
                            left_turn,
                            line_types,
                            weight,
                            LaneType.INTERSECTION,
                            lane_priority,
                            width=lane_width
                        )
                        
                    else:
                        continue  # Skip invalid direction combinations
                

                    road_desc[self.PathType.CIRCULAR].append(path)
        
        # Add the constructed roads to the network
        self.add_multiple_paths(road_desc)
        
    def add_roundabout(self):
        """
        Naming convention
        -----------------
            This framework, and subsequently this method, use the naming convention<br>
            ``<type>-<id<:<direction>-<in|out>[:lane_index]``. This method will thus generate the ``in|out`` nodes for each direction.
            <br><br>
            - ``type`` refers to the type of the road, e.g. "``I``" for "intersection" or "``T``" for "turn".<br>
            - ``id`` refers to the id of the type, e.g. the first intersection is ``I-1`` and the second is ``I-2``.<br>
            - ``direction`` refers to the cardinal traversal direction of the road which are about to be made.<br>
            - ``in|out`` refers to the node either being an ``in`` node or ``out`` node. An ``In`` node is the<br>
                node where you enter the road and the ``out`` node is the node where you exit the road.<br>
                E.g. ``in`` nodes in an intersection are the ones which enters the intersection<br>
                while the ``out`` nodes are the ones which exits the intersection.<br>
            - ``:lane_index`` refers to the lane's index such that it is possible<br>
                to have multiple lanes from between the same nodes.<br>
                The ``:``is a separator between the direction part and the lane indexation<br>
                The ``:lane_index`` will be removed from the node name when building<br>
                such that the network will know that the lanes are connected and<br>
                it is possible to move between them. However, the NetworkBuilder will<br>
                still store in its dictionary of nodes and paths the ``:lane_index``<br>
                you provide - thus it is <b><i>only</i></b> when building the network it removes the lane index
            
        Description
        -----------
            Adds an intersection to the road network, generating and configuring<br>
            incoming and outgoing lanes based on the specified direction,<br>
            priority, and lane width. This method establishes both straight and<br>
            circular paths between directions.

        Parameters
        ----------
            intersection_name: str
                A unique identifier for the intersection.
                
            ingoing_roads: dict[CardinalDirection, Vector]
                Maps directions (North, South, East, West) to their respective<br>
                vectors, which define the positions of ingoing roads.
            
            priority: PathPriority
                The prioritized lane direction (e.g., North-South or East-West).
            
            lane_width: float
                Width of each lane, with a default value from AbstractLane.

        Mapping of an ingoing point to other ingoing points
        ----------------------
        This asumes that you <b>do not</b> modify the lane width.<br>
        
        <b>North-in</b> -> <b>south-in</b> mapping: ``[+4, +12]`` <br>
        <b>North-in</b> -> <b>West-in</b> mapping: ``[-4, +8]`` <br>
        <b>North-in</b> -> <b>East-in</b> mapping: ``[+8, +4]`` <br><br>

        <b>South-in</b> -> <b>North-in</b> mapping: ``[-4, -12]`` <br>
        <b>South-in</b> -> <b>West-in</b> mapping: ``[-8, -4]`` <br>
        <b>South-in</b> -> <b>East-in</b> mapping: ``[+4, -8]`` <br><br>
        
        <b>West-in</b> -> <b>East-in</b> mapping: ``[+12, -4]`` <br>
        <b>West-in</b> -> <b>North-in</b> mapping: ``[+4, -8]`` <br>
        <b>West-in</b> -> <b>south-in</b> mapping: ``[+8, +4]`` <br><br>
        
        <b>East-in</b> -> <b>West-in</b> mapping: ``[-12, +4]`` <br>
        <b>East-in</b> -> <b>North-in</b> mapping: ``[-8, -4]`` <br>
        <b>East-in</b> -> <b>south-in</b> mapping: ``[-4 , +8]`` <br>

        Example
        -------
        ```python
        nb = NetworkBuilder()
        nb.add_intersection(
            "I-2",
            {
                net_builder.CardinalDirection.NORTH : [64, -92],
                net_builder.CardinalDirection.SOUTH : [68, -80],
                net_builder.CardinalDirection.WEST  : [60, -84],
            },
            net_builder.PathPriority.NORTH_SOUTH
        )
        # This will generate the following nodes:
        # "I-2:n-in"
        # "I-2:s-in"
        # "I-2:w-in"
        # "I-2:n-out"
        # "I-2:s-out"
        # "I-2:w-out"
        ```
        
        Assumptions
        -----------
            The directions are expected to be the cardinal directions North, South,<br>
            East, and West. This method manages the road layout to handle<br>
            straight and turning paths based on priorities and intersection geometry.
        """
        print("Not implemented")
        
    def _build_straight_path(self, path: StraightPath) -> tuple[str, str, StraightLane, int, LaneType]:
        """
        Description
        -----------
            <b>Private method</b> which extracts the paths in the NetworkBuilder and<br>
            converts it to the correct information needed for a lane.

        Parameters
        ----------
        path : StraightPath
            A StraightPath which should connect two nodes.
        Returns
        -------
            ``tuple[str, str, StraightLane, int, LaneType]`` <br>
            Returns the needed information for the ``RoadNetwork.add_lane(...)`` method
        """
        return (
            path.from_node_id,
            path.to_node_id,
            StraightLane(
                self._nodes[path.from_node_id],
                self._nodes[path.to_node_id],
                path.width,
                path.line_types,
                path.forbidden,
                path.speed_limit,
                path.priority,
            ),
            path.weight,
            path.lane_type
        )
        
    def _build_circular_path(self, path: CircularPath) -> tuple[str, str, CircularLane, int, LaneType]:
        """
        Description
        -----------
            <b>Private method</b> which extracts the paths in the NetworkBuilder and<br>
            converts it to the correct information needed for a lane.

        Parameters
        ----------
        path : CircularPath
            A CircluarPath which should connect two nodes.
            
        Returns
        -------
            ``tuple[str, str, CircularLane, int, LaneType]`` <br>
            Returns the needed information for the ``RoadNetwork.add_lane(...)`` method
        """

        
        end_phase: float = path.start_phase - path.get_phase(
            self._nodes[path.from_node_id],
            self._nodes[path.to_node_id],
            path.radius,
            path.clockwise
        )
        
        if path.clockwise:
            end_phase += 360
        
        center: Vector = path.get_center(
            self._nodes[path.from_node_id],
            self._nodes[path.to_node_id],
            path.radius,
            path.radius,
            path.clockwise
        )
        
        return (
            path.from_node_id,
            path.to_node_id,
            CircularLane(
                center,
                path.radius,
                np.deg2rad(path.start_phase),
                np.deg2rad(end_phase),
                path.clockwise,
                path.width,
                path.line_types,
                path.forbidden,
                path.speed_limit,
                path.priority
            ),
            path.weight,
            path.lane_type
        )
    
    def _build_sine_path(self, path: SinePath) -> tuple[str, str, SineLane, int, LaneType]:
        """
        Description
        -----------
            <b>Private method</b> which extracts the paths in the NetworkBuilder and<br>
            converts it to the correct information needed for a lane.

        Parameters
        ----------
        path : SinePath
            A SinePath which should connect two nodes.
        
        Returns
        -------
            ``tuple[str, str, SineLane, int, LaneType]`` <br>
            Returns the needed information for the ``RoadNetwork.add_lane(...)`` method
        """
        
        return (
            path.from_node_id,
            path.to_node_id,
            SineLane(),
            path.weight,
            path.lane_type
        )
    
    def build_paths(self, road_network: RoadNetwork):
        """
        Description
        -----------
            Adds the content of the NetworkBuilder to the passed RoadNetwork.<br>
            The content refers to the nodes and paths which gets translated into<br>
            vertices and edges (nodes and lanes).

        Parameters
        ----------
        road_network : RoadNetwork
            The road network to add the nodes and paths to.
        """
        net: list[tuple[str, str, AbstractLane]] = []

        # Mapping path types to their respective build methods
        build_methods = {
            self.PathType.STRAIGHT: self._build_straight_path,
            self.PathType.CIRCULAR: self._build_circular_path,
            self.PathType.SINE: self._build_sine_path
        }

        # Iterate through each path type and description
        for path_type, build_method in build_methods.items():
            for path in self._path_description[path_type]:
                net.append(build_method(path))

        # Add each lane to the road network
        for from_id, to_id, lane, weight, lane_type in net:
            road_network.add_lane(
                re.sub(r":\d+$", "", from_id),
                re.sub(r":\d+$", "", to_id),
                lane,
                weight=weight,
                lane_type=lane_type
            )