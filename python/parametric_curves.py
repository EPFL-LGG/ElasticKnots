import numpy as np
from numpy import sin, cos

# ------------------------------------------------------------------------
# Parametric curves
# ------------------------------------------------------------------------

def generate_curve(parametric_curve, npts, ti=0, tf=2*np.pi, duplicated_points=2):
    t = np.linspace(ti, tf, npts-1)  # endpoints included
    t = np.append(t, t[1:duplicated_points])
    x, y, z = parametric_curve(t)
    curve = np.zeros((npts, 3))
    curve[:, 0] = x
    curve[:, 1] = y
    curve[:, 2] = z
    return curve


def circle(r):
    x = lambda t: r * cos(t)
    y = lambda t: r * sin(t)
    z = lambda t: 0 * t
    return lambda t: [x(t), y(t), z(t)]


def ellipse(a, b):
    x = lambda t: a * cos(t)
    y = lambda t: b * sin(t)
    z = lambda t: 0 * t
    return lambda t: [x(t), y(t), z(t)]

    
def trefoil(r):
    x = lambda t: r * (sin(t) + 2*sin(2*t))
    y = lambda t: r * (cos(t) - 2*cos(2*t))
    z = lambda t: r * (sin(3*t))
    return lambda t: [x(t), y(t), z(t)]


def figure_eight(r):
    x = lambda t: r * (2 + cos(2*t))*cos(3*t)
    y = lambda t: r * (2 + cos(2*t))*sin(3*t)
    z = lambda t: r * sin(4*t)
    return lambda t: [x(t), y(t), z(t)]


def torus_knot(p, q, a=2, r=1):
    x = lambda t: r * (cos(p*t) * (a + cos(q*t)))
    y = lambda t: r * (sin(p*t) * (a + cos(q*t)))
    z = lambda t: r * (sin(q*t))
    return lambda t: [x(t), y(t), z(t)]