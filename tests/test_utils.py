import numpy as np

from highway_env.utils import rotated_rectangles_intersect


def test_rotated_rectangles_intersect():
    assert rotated_rectangles_intersect(([12.86076812, 28.60182391], 5.0, 2.0, -0.4675779906495494),
                                        ([9.67753944, 28.90585412], 5.0, 2.0, -0.3417019364473201))
    assert rotated_rectangles_intersect(([0, 0], 2, 1, 0), ([0, 1], 2, 1, 0))
    assert not rotated_rectangles_intersect(([0, 0], 2, 1, 0), ([0, 2.1], 2, 1, 0))
    assert not rotated_rectangles_intersect(([0, 0], 2, 1, 0), ([1, 1.1], 2, 1, 0))
    assert rotated_rectangles_intersect(([0, 0], 2, 1, np.pi/4), ([1, 1.1], 2, 1, 0))
