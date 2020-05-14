#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created by Roman Polishchenko at 11.05.2020
3rd course, computer mathematics
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
from sympy.matrices import Matrix
from sympy.core.numbers import Zero
from sympy import diff
from sympy.core.expr import Expr

PARAM = 't'
EPS = 0.01
Pi = np.pi


class ParametricCurve3D:
    def __init__(self, x_func: str, y_func: str, z_func: str, t_range: tuple, param=PARAM) -> None:
        """ Create a 3d curve with r(t) = (x(t), y(t), z(t)) parametrization.

        :params x_func, y_func, z_func: must depend on param (default: t)
        :param t_range: variability range for parameter
        :param param: string that represents the functions' parameter
        """
        assert len(t_range) == 2, 't_range must have len = 2'
        assert all(isinstance(elem, (float, int)) for elem in t_range), 't_range parameters must be numbers'

        self._t = symbols(param)
        self._t_lower = float(t_range[0])
        self._t_upper = float(t_range[1])

        self._x_func = parse_expr(x_func)
        self._y_func = parse_expr(y_func)
        self._z_func = parse_expr(z_func)

        # create vectorized functions for x, y, z
        self._x = lambdify(self._t, self._x_func, 'numpy')
        self._y = lambdify(self._t, self._y_func, 'numpy')
        self._z = lambdify(self._t, self._z_func, 'numpy')

        mpl.rcParams['legend.fontsize'] = 10
        fig = plt.figure()
        self._ax = fig.gca(projection='3d')

    def __repr__(self):
        return 'r({t}) = ({x}, {y}, {z})'.format(t=self._t, x=self._x_func, y=self._y_func, z=self._z_func)

    def __str__(self):
        return 'r({t}) = ({x}, {y}, {z})'.format(t=self._t, x=self._x_func, y=self._y_func, z=self._z_func)

    def plot(self, neighborhood=None, num=1000) -> None:
        """ Plot the curve on graphic.

            :param neighborhood: if given, plot in the neighborhood of the point
            :param num: number of points in linspace, e.g. bigger num -> detailed plot
        """
        if not neighborhood:
            t = np.linspace(self._t_lower, self._t_upper, num)
        else:
            assert len(neighborhood) == 2 and \
                   isinstance(neighborhood[0], (float, int)) and \
                   isinstance(neighborhood[1], (float, int)), 'neighborhood must contain two numbers'
            t = np.linspace(neighborhood[0], neighborhood[1], num)
        x = self._x(t)
        y = self._y(t)
        z = self._z(t)

        self._ax.plot3D(x, y, z, label=str(self))
        self._ax.legend()
        set_axes_equal(self._ax)
        plt.show()

    def tangent_vector(self, t: float, plot=False) -> np.ndarray:
        """ Unit vector tangent to the curve, pointing in the direction of motion at point t.
            τ = r'(t) / |r'(t)|

            :param t: given point
            :param plot: if true - add tangent unit vector to the plot, when self.plot function is called
        """
        self._validate_t(t)
        r_der = self._get_derivative(t)
        module = ParametricCurve3D._module(r_der)
        vector = np.array(list(map(lambda x: x / module, r_der)))
        if plot:
            origin = self._at_point(t)
            self._plt_vector(origin, vector, color='r', label='tangent unit vector')
        return vector

    def normal_vector(self, t: float, plot=False) -> np.ndarray:
        """ Normal unit vector at point t.
            ν = [ß, τ]

            :param t: given point
            :param plot: if true - add normal unit vector to the plot, when self.plot function is called
        """
        self._validate_t(t)
        beta = self.binormal_vector(t)
        tau = self.tangent_vector(t)
        nu = np.cross(beta, tau)
        if plot:
            origin = self._at_point(t)
            self._plt_vector(origin, nu, color='g', label='normal unit vector')
        return nu

    def binormal_vector(self, t: float, plot=False) -> np.ndarray:
        """ Binormal unit vector at point t, cross product of T and N.
            ß = [r'(t), r''(t)] / |[r'(t), r''(t)]|

            :param t: given point
            :param plot: if true - add binormal unit vector to the plot, when self.plot function is called
        """
        self._validate_t(t)
        r_der = self._get_derivative(t)
        r_der_2 = self._get_derivative(t, order=2)
        product = np.cross(r_der, r_der_2)
        module = ParametricCurve3D._module(product)
        vector = np.array(list(map(lambda x: x / module, product)))
        if plot:
            origin = self._at_point(t)
            self._plt_vector(origin, vector, color='b', label='binormal unit vector')
        return vector

    # стична
    def osculating_plane(self, t: float) -> Expr:
        """ Tangent plane to the curve at point t.

            :param t: given point
        """
        self._validate_t(t)
        r_der = self._get_derivative(t)
        r_der_2 = self._get_derivative(t, order=2)
        p = self._at_point(t)
        return ParametricCurve3D._find_plane(p, r_der, r_der_2)

    # нормальна
    def normal_plane(self, t: float) -> Expr:
        """ Normal plane at point t.

            :param t: given point
        """
        self._validate_t(t)
        beta = self.binormal_vector(t)
        nu = self.normal_vector(t)
        p = self._at_point(t)
        return ParametricCurve3D._find_plane(p, nu, beta)

    # спрямна
    def reference_plane(self, t: float) -> Expr:
        """ Binormal plane at point t.

            :param t: given point
        """
        self._validate_t(t)
        beta = self.binormal_vector(t)
        tau = self.tangent_vector(t)
        p = self._at_point(t)
        return ParametricCurve3D._find_plane(p, tau, beta)

    def curvature(self, t: float) -> float:
        """ Curvature at point t – the amount by which a curve deviates from being a straight line.
            k = |[r'(t), r''(t)]| / |r'(t)|**3

            :param t: given point
        """
        self._validate_t(t)
        r_der = self._get_derivative(t)
        r_der_2 = self._get_derivative(t, 2)
        product = np.cross(r_der, r_der_2)
        curvature = ParametricCurve3D._module(product) / ParametricCurve3D._module(r_der) ** 3
        return curvature

    def torsion(self, t: float) -> float:
        """ The torsion of a curve measures how sharply it is twisting out of the plane of curvature.
            kappa = (r'(t), r''(t), r'''(t)) / |[r'(t), r''(t)]|**2

            :param t: given point
        """
        self._validate_t(t)
        r_der = self._get_derivative(t)
        r_der_2 = self._get_derivative(t, 2)
        r_der_3 = self._get_derivative(t, 3)
        numerator = Matrix([r_der, r_der_2, r_der_3]).det()
        denominator = ParametricCurve3D._module(np.cross(r_der, r_der_2)) ** 2
        torsion = numerator / denominator
        return torsion

    def osculating_circle(self, t: float) -> tuple:
        """ Osculating circle to the curve at the point t: intersection of osculating sphere and plane.
            if curve is line -> return (None, None), because there is no osculating circle

            :param t: given point
            :return: tuple contains equations for osculating sphere and plane.
        """
        self._validate_t(t)
        kappa = self.curvature(t)
        # if kappa = 0 -> curve is line and there is no osculating circle
        if kappa == 0:
            return None, None
        radius = 1 / kappa
        tangent_vector = self.tangent_vector(t)
        r0 = self._at_point(t)
        r_vector = r0 + radius*tangent_vector
        x, y, z = symbols('x, y, z')
        osculating_sphere = (x - r_vector[0])**2 + (y - r_vector[1])**2 + (z - r_vector[2])**2 - radius**2
        osculating_plane = self.osculating_plane(t)
        return osculating_sphere, osculating_plane

    # ======================================= Helper methods ======================================= #
    @staticmethod
    def _module(vector) -> float:
        """ Helper method to calculate module of given vector of numbers"""
        assert len(vector) == 3, 'vector length must be 3'
        assert all(isinstance(elem, (float, int)) for elem in vector), 'vector elements must be numbers'
        module = 0
        for f in vector:
            module += f**2
        return np.sqrt(module)

    def _plt_vector(self, origin, vector, color='b', label=None) -> None:
        """ Helper method to add vector from origin point to self._ax figure """
        assert len(origin) == 3, 'origin must contain 3 coordinates'
        assert len(vector) == 3, 'vector must contain 3 coordinates'
        assert all(isinstance(elem, (float, int)) for elem in origin), 'origin elements must be numbers'
        assert all(isinstance(elem, (float, int)) for elem in vector), 'vector elements must be numbers'

        self._ax.quiver(*origin, *vector, color=color, label=label)

    def _at_point(self, t) -> tuple:
        """ Helper method to calculate r(t) at given point t """
        self._validate_t(t)
        return (float(self._x_func.subs(self._t, t)),
                float(self._y_func.subs(self._t, t)),
                float(self._z_func.subs(self._t, t)))

    def _validate_t(self, t) -> None:
        """ Helper method to check weather """
        assert self._t_lower <= t <= self._t_upper, 't not in the t_range'

    @staticmethod
    def _find_plane(p, vector1, vector2) -> Expr:
        """ Helper method tp find plane that fits point p and two non-collinear vectors """
        x, y, z = symbols('x, y, z')
        xyz = np.array([x, y, z])
        matrix = Matrix([xyz - p, vector1, vector2])
        plane = matrix.det()
        if plane == Zero:
            raise AssertionError('vector1 and vector2 must be non-collinear')
        return plane

    def _get_derivative(self, t, order=1) -> np.ndarray:
        """ Helper method to find order'th derivative of r at given point t"""
        return np.array([float(diff(self._x_func, self._t, order).subs(self._t, t)),
                         float(diff(self._y_func, self._t, order).subs(self._t, t)),
                         float(diff(self._z_func, self._t, order).subs(self._t, t))])


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# testing
if __name__ == '__main__':
    # test 1
    curve = ParametricCurve3D('sin(t)', 'cos(t)', 'tan(t)', (0, 5))
    point = Pi / 4
    print('For curve {} at point {}:'.format(curve, point))
    print('\ttangent unit vector: {}'.format(curve.tangent_vector(point, plot=True)))
    print('\tnormal unit vector: {}'.format(curve.normal_vector(point, plot=True)))
    print('\tbinormal unit vector: {}'.format(curve.binormal_vector(point, plot=True)))
    curve.plot(neighborhood=(Pi / 12, Pi / 3))

    # test 2
    curve2 = ParametricCurve3D('2*(t-sin(t))', '2*(t-cos(t))', '8*cos(t/2)', (-Pi, Pi))
    point2 = 0
    print('\nFor curve {} at point {}:'.format(curve2, point2))
    print('\tcurvature: {}'.format(curve2.curvature(point2)))
    print('\ttorsion: {}'.format(curve2.torsion(point2)))
    osc_sphere, osc_plane = curve2.osculating_circle(point2)
    print('\t# osculating circle <=> intersection of osculating sphere and osculating plane')
    print('\tosculating sphere: {} = 0'.format(osc_sphere))
    print('\tosculating plane: {} = 0'.format(osc_plane))

    # test 3
    curve3 = ParametricCurve3D('t', 't**3', 't**2+1', (0, 2))
    point3 = 1
    print('\nFor curve {} at point {}:'.format(curve3, point3))
    print('\tosculating plane: {} = 0'.format(curve3.osculating_plane(point3)))
    print('\tnormal plane: {} = 0'.format(curve3.normal_plane(point3)))
    print('\treference plane: {} = 0'.format(curve3.reference_plane(point3)))
