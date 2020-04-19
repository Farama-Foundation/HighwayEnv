import importlib
import numpy as np

EPSILON = 0.01


def constrain(x, a, b):
    return np.minimum(np.maximum(x, a), b)


def not_zero(x):
    if abs(x) > EPSILON:
        return x
    elif x > 0:
        return EPSILON
    else:
        return -EPSILON


def wrap_to_pi(x):
    return ((x+np.pi) % (2*np.pi)) - np.pi


def point_in_rectangle(point, rect_min, rect_max):
    """
        Check if a point is inside a rectangle
    :param point: a point (x, y)
    :param rect_min: x_min, y_min
    :param rect_max: x_max, y_max
    """
    return rect_min[0] <= point[0] <= rect_max[0] and rect_min[1] <= point[1] <= rect_max[1]


def point_in_rotated_rectangle(point, center, length, width, angle):
    """
        Check if a point is inside a rotated rectangle
    :param point: a point
    :param center: rectangle center
    :param length: rectangle length
    :param width: rectangle width
    :param angle: rectangle angle [rad]
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return point_in_rectangle(ru, [-length/2, -width/2], [length/2, width/2])


def point_in_ellipse(point, center, angle, length, width):
    """
        Check if a point is inside an ellipse
    :param point: a point
    :param center: ellipse center
    :param angle: ellipse main axis angle
    :param length: ellipse big axis
    :param width: ellipse small axis
    """
    c, s = np.cos(angle), np.sin(angle)
    r = np.matrix([[c, -s], [s, c]])
    ru = r.dot(point - center)
    return np.sum(np.square(ru / np.array([length, width]))) < 1


def rotated_rectangles_intersect(rect1, rect2):
    """
        Do two rotated rectangles intersect?
    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    """
    return has_corner_inside(rect1, rect2) or has_corner_inside(rect2, rect1)


def has_corner_inside(rect1, rect2):
    """
        Check if rect1 has a corner inside rect2
    :param rect1: (center, length, width, angle)
    :param rect2: (center, length, width, angle)
    """
    (c1, l1, w1, a1) = rect1
    (c2, l2, w2, a2) = rect2
    c1 = np.array(c1)
    l1v = np.array([l1/2, 0])
    w1v = np.array([0, w1/2])
    r1_points = np.array([[0, 0],
                          - l1v, l1v, -w1v, w1v,
                          - l1v - w1v, - l1v + w1v, + l1v - w1v, + l1v + w1v])
    c, s = np.cos(a1), np.sin(a1)
    r = np.array([[c, -s], [s, c]])
    rotated_r1_points = r.dot(r1_points.transpose()).transpose()
    return any([point_in_rotated_rectangle(c1+np.squeeze(p), c2, l2, w2, a2) for p in rotated_r1_points])


def do_every(duration, timer):
    return duration < timer


def remap(v, x, y):
    return y[0] + (v-x[0])*(y[1]-y[0])/(x[1]-x[0])


def class_from_path(path):
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object
