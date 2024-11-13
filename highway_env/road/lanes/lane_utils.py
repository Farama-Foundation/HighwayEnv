from enum import Enum

class LineType:
    """A lane side line type."""
    NONE = 0
    STRIPED = 1
    CONTINUOUS = 2
    CONTINUOUS_LINE = 3

class LaneType(Enum):
    INTERSECTION = 'intersection'
    ROUNDABOUT = 'roundabout'
    ROAD_ENTRY = 'road-entry'
    ROAD_EXIT = 'road-exit'
    HIGHWAY = 'highway'
    ROAD = 'road'
