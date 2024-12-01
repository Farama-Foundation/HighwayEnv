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
    HIGHWAY_ENTRY = 'highway-entry'
    HIGHWAY_EXIT = 'highway-exit'
    HIGHWAY = 'highway'
    ROAD = 'road'
    CITY_ROAD = 'city-road'
