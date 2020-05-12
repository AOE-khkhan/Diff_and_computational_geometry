#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created by Roman Polishchenko at 11.05.2020
3 course, comp math
Taras Shevchenko National University of Kyiv
email: roma.vinn@gmail.com
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sympy.core.symbol import symbols
from sympy.parsing.sympy_parser import parse_expr
from sympy.utilities import lambdify

PARAM = 't'
EPS = 0.01
PI = np.pi

class Curve3D:
    def __init__(self, x_func: str, y_func: str, z_func: str, t_range: tuple, param=PARAM):
        """
        x_func, y_func, z_func must be with parameter t.
        """
        assert len(t_range) == 2, 't_range must have len = 2'
        assert isinstance(t_range[0], (float, int)) and isinstance(t_range[1], (float, int)), \
            't_range parameters must be numbers'

        self._t = symbols(param)
        self._t_lower = float(t_range[0])
        self._t_upper = float(t_range[1])

        self._x_func = parse_expr(x_func)
        self._y_func = parse_expr(y_func)
        self._z_func = parse_expr(z_func)

        self._x = lambdify(self._t, self._x_func, 'numpy')
        self._y = lambdify(self._t, self._y_func, 'numpy')
        self._z = lambdify(self._t, self._z_func, 'numpy')

        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        self._ax = fig.gca(projection='3d')

    def plot(self, neighborhood=None):
        """
        Plot the curve on graphic.
        """
        if not neighborhood:
            t = np.linspace(self._t_lower, self._t_upper, 1000)
        else:
            assert len(neighborhood) == 2 and \
                   isinstance(neighborhood[0], (float, int)) and \
                   isinstance(neighborhood[1], (float, int)), 'neighborhood must contain two numbers'
            t = np.linspace(neighborhood[0], neighborhood[1], 1000)
        x = self._x(t)
        y = self._y(t)
        z = self._z(t)

        self._ax.plot(x, y, z, label='r({t}) = ({x}, {y}, {z})'.format(t=self._t,
                                                                       x=self._x_func,
                                                                       y=self._y_func,
                                                                       z=self._z_func))
        self._ax.legend()
        plt.show()

    def tangent_vector(self, t: float, plot=False):
        """
        T – Unit vector tangent to the curve, pointing in the direction of motion at point t.
        """
        r_der = (float(self._x_func.diff().subs(self._t, t)),
                 float(self._y_func.diff().subs(self._t, t)),
                 float(self._z_func.diff().subs(self._t, t)))
        module = Curve3D._module(r_der)
        vector = tuple(map(lambda x: x / module, r_der))
        if plot:
            origin = self._at_point(t)
            self._plt_vector(origin, vector, color='r')
        return vector

    # TODO review
    def normal_vector(self, t: float, plot=False):
        """
        N - Normal unit vector at point t.
        """
        beta = self.binormal_vector(t)
        tau = self.tangent_vector(t)
        nu = np.cross(beta, tau)
        if plot:
            origin = self._at_point(t)
            self._plt_vector(origin, nu, color='g')
        return nu

    # TODO review
    def binormal_vector(self, t: float, plot=False):
        """
        B - Binormal unit vector at point t, cross product of T and N.
        """
        r_der = (float(self._x_func.diff().subs(self._t, t)),
                 float(self._y_func.diff().subs(self._t, t)),
                 float(self._z_func.diff().subs(self._t, t)))
        r_der_2 = (float(self._x_func.diff().diff().subs(self._t, t)),
                   float(self._y_func.diff().diff().subs(self._t, t)),
                   float(self._z_func.diff().diff().subs(self._t, t)))
        product = np.cross(r_der, r_der_2)
        module = Curve3D._module(product)
        vector = tuple(map(lambda x: x / module, product))
        if plot:
            origin = self._at_point(t)
            self._plt_vector(origin, vector, color='b')
        return vector

    def tangent_plane(self, t: float):
        """
        Tangent plane to the curve at point t.
        """
        pass

    def normal_plane(self, t: float):
        """
        Normal plane at point t.
        """
        pass

    def binormal_plane(self, t: float):
        """
        Binormal plane at point t.
        """
        pass

    def curvature(self, t: float):
        """
        Curvature at point t – the amount by which a curve deviates from being a straight line.
        """
        pass

    def torsion(self, t: float):
        """
        The torsion of a curve measures how sharply it is twisting out of the plane of curvature.
        :param t:
        :return:
        """
        pass

    def adjacent_circle(self, t: float):
        pass

    @staticmethod
    def _module(vector):
        """Helper method to calculate module of given vector"""
        assert len(vector) == 3, 'vector length must be 3'
        module = 0
        for f in vector:
            module += f**2
        return np.sqrt(module)

    def _plt_vector(self, origin, vector, color='b'):
        assert len(origin) == 3, 'origin must contain 3 coordinates'
        assert len(vector) == 3, 'vector must contain 3 coordinates'
        origin = [origin[0]], [origin[1]], [origin[2]]
        vector = [vector[0]], [vector[1]], [vector[2]]
        x, y, z, u, v, w = zip(*[origin + vector])
        self._ax.quiver(x, y, z, u, v, w, color=color)

    def _at_point(self, t):
        return (float(self._x_func.subs(self._t, t)),
                float(self._y_func.subs(self._t, t)),
                float(self._z_func.subs(self._t, t)))


# testing
if __name__ == '__main__':
    curve = Curve3D('sin(t)', 'cos(t)', 'tan(t)', (0, 5))
    point = PI / 4
    tangent_vector = curve.tangent_vector(point, plot=True)
    normal_vector = curve.normal_vector(point, plot=True)
    binormal_vector = curve.binormal_vector(point, plot=True)
    curve.plot(neighborhood=(EPS-PI/2, PI/2-EPS))
    print(tangent_vector, normal_vector, binormal_vector)
