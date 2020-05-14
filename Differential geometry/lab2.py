#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created by Roman Polishchenko at 14.05.2020
3rd course, computer mathematics
Taras Shevchenko National University of Kyiv
email: roma.vinn@gmail.com
"""

import numpy as np
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.symbol import symbols, Symbol
from sympy.core.expr import Expr
from sympy.matrices import Matrix
from sympy import diff, Eq, Function, solve
from sympy.core.numbers import Zero
from time import time


class GeneralCurve3D:
    def __init__(self, f: str, g: str, params='x y z') -> None:
        """ Create a 3D curve defined by intersection of 2 surfaces

        :param f: surface f = F(x, y, z)
        :param g: surface g = G(x, y, z)
        """
        self._f = parse_expr(f)
        self._g = parse_expr(g)
        self._x, self._y, self._z = symbols(params)
        self._der1 = dict()
        self._der2 = dict()

    def __repr__(self):
        """ String representation """
        return "F: {}\nG: {}".format(self._f, self._g)

    def __str__(self):
        """ String representation """
        return "F: {}\nG: {}".format(self._f, self._g)

    def find_derivatives(self, p: tuple) -> tuple:
        """ Check conditions of implicit function theorem and return r'(p), r''(p)
                                (теореми про неявну функцію)

            1) We can't check, so we suppose that f, g є C**oo
            2) Check if F(p) = G(p) = 0
            3) Find such variable z, so d(f, g) / d(x, y) at point p != 0
                (check all possible variables)

            :param p: point consist of 3 numbers
            :return dr1, dr2 - first and second derivatives of r wrt f, g
        """
        # caching – if calculated before –> just return it
        if self._der1.get(str(p)) is not None:
            return self._der1.get(str(p)), self._der2.get(str(p))

        subs_dict = {self._x: p[0], self._y: p[1], self._z: p[2]}
        cur_params = [self._x, self._y, self._z]
        x, y, z = cur_params
        found = False
        for _ in range(3):
            x, y, z = cur_params
            if self._jacobian(x, y, p) != 0:
                found = True
                break
            cur_params = cur_params[-1:] + cur_params[:2]
        assert found is True, 'there is no not-null jacobian'

        f = Function('f')(z)
        g = Function('g')(z)
        df1, dg1, df2, dg2 = symbols('df1, dg1, df2, dg2')
        # ========================== find first derivatives ========================== #
        eq1 = self._f.subs({x: f, y: g}).diff(z).subs(subs_dict)
        eq2 = self._g.subs({x: f, y: g}).diff(z).subs(subs_dict)

        def _sub_1(str_eq: str) -> str:
            str_eq = str_eq.replace(f'f({subs_dict[z]})', '(' + str(subs_dict[x]) + ')')
            str_eq = str_eq.replace(f'g({subs_dict[z]})', '(' + str(subs_dict[y]) + ')')
            str_eq = str_eq.replace(f'Subs(Derivative(f({z}), {z}), ({z},), ({subs_dict[z]},))', '(df1)')
            str_eq = str_eq.replace(f'Subs(Derivative(g({z}), {z}), ({z},), ({subs_dict[z]},))', '(dg1)')
            return str_eq

        str_eq1 = _sub_1(str(eq1))
        str_eq2 = _sub_1(str(eq2))

        eq1 = Eq(parse_expr(str_eq1), 0)
        eq2 = Eq(parse_expr(str_eq2), 0)
        roots = solve([eq1, eq2])
        df1 = roots[df1]
        dg1 = roots[dg1]

        correct_order = [self._x, self._y, self._z]
        dr1 = np.array([0, 0, 0])
        dr1[correct_order.index(x)] = df1
        dr1[correct_order.index(y)] = dg1
        dr1[correct_order.index(z)] = 1
        # ========================= find second derivatives ========================== #
        eq1 = self._f.subs({x: f, y: g}).diff(z, 2).subs(subs_dict)
        eq2 = self._g.subs({x: f, y: g}).diff(z, 2).subs(subs_dict)

        def _sub_2(str_eq: str) -> str:
            str_eq = str_eq.replace('df1', '(' + str(float(df1)) + ')')
            str_eq = str_eq.replace('dg1', '(' + str(float(dg1)) + ')')
            str_eq = str_eq.replace(f'Subs(Derivative(f({z}), {z}, {z}), ({z},), ({subs_dict[z]},))', '(df2)')
            str_eq = str_eq.replace(f'Subs(Derivative(g({z}), {z}, {z}), ({z},), ({subs_dict[z]},))', '(dg2)')
            return str_eq

        str_eq1 = _sub_2(_sub_1(str(eq1)))
        str_eq2 = _sub_2(_sub_1(str(eq2)))

        eq1 = Eq(parse_expr(str_eq1), 0)
        eq2 = Eq(parse_expr(str_eq2), 0)
        roots = solve([eq1, eq2])
        df2 = roots[df2]
        dg2 = roots[dg2]

        dr2 = np.array([0, 0, 0])
        dr2[correct_order.index(x)] = df2
        dr2[correct_order.index(y)] = dg2
        dr2[correct_order.index(z)] = 0

        # cache it
        self._der1[str(p)] = dr1
        self._der2[str(p)] = dr2
        return dr1, dr2

    def _jacobian(self, x: Symbol, y: Symbol, p: tuple) -> float:
        subs_dict = {self._x: p[0], self._y: p[1], self._z: p[2]}

        matrix = Matrix(
            [[diff(self._f, x).subs(subs_dict), diff(self._f, y).subs(subs_dict)],
             [diff(self._g, x).subs(subs_dict), diff(self._g, y).subs(subs_dict)]]
        )

        return matrix.det()

    def tangent_vector(self, p: tuple) -> np.ndarray:
        """ Unit vector tangent to the curve, pointing in the direction of motion at point p.
            τ = r'(p) / |r'(p)|

            :param p: given point
        """
        der1, der2 = self.find_derivatives(p)
        module = np.linalg.norm(der1)
        vector = der1 / module
        return vector

    def normal_vector(self, p: tuple) -> np.ndarray:
        """ Normal unit vector at point p.
            ν = [ß, τ]

            :param p: given point
        """
        der1, der2 = self.find_derivatives(p)
        product = np.cross(der1, der2)
        nu = np.cross(product, der1)
        module = np.linalg.norm(nu)
        vector = nu / module
        return vector

    def binormal_vector(self, p: tuple) -> np.ndarray:
        """ Binormal unit vector at point p, cross product of T and N.
            ß = [r'(p), r''(p)] / |[r'(p), r''(p)]|

            :param p: given point
        """
        der1, der2 = self.find_derivatives(p)
        product = np.cross(der1, der2)
        module = np.linalg.norm(product)
        vector = product / module
        return vector

    # стична
    def osculating_plane(self, p: tuple) -> Expr:
        """ Tangent plane to the curve at point p.

            :param p: given point
        """
        der1, der2 = self.find_derivatives(p)
        return GeneralCurve3D._find_plane(p, der1, der2)

    # нормальна
    def normal_plane(self, p: tuple) -> Expr:
        """ Normal plane at point p.

            :param p: given point
        """
        beta = self.binormal_vector(p)
        nu = self.normal_vector(p)
        return GeneralCurve3D._find_plane(p, nu, beta)

    # спрямна
    def reference_plane(self, p: tuple) -> Expr:
        """ Binormal plane at point p.

            :param p: given point
        """
        beta = self.binormal_vector(p)
        tau = self.tangent_vector(p)
        return GeneralCurve3D._find_plane(p, tau, beta)

    def curvature(self, p: tuple) -> float:
        """ Curvature at point p – the amount by which a curve deviates from being a straight line.
            k = |[r'(p), r''(p)]| / |r'(p)|**3

            :param p: given point
        """
        der1, der2 = self.find_derivatives(p)
        product = np.cross(der1, der2)
        curvature = np.linalg.norm(product) / np.linalg.norm(der1) ** 3
        return curvature

    def osculating_circle(self, p: tuple) -> tuple:
        """ Osculating circle to the curve at the point t: intersection of osculating sphere and plane.
            if curve is line -> return (None, None), because there is no osculating circle

            :param p: given point
            :return: tuple contains equations for osculating sphere and plane.
        """
        kappa = self.curvature(p)
        # if kappa = 0 -> curve is line and there is no osculating circle
        if kappa == 0:
            return None, None
        radius = 1 / kappa
        tangent_vector = self.tangent_vector(p)
        r_vector = p + radius*tangent_vector
        x, y, z = symbols('x, y, z')
        osculating_sphere = (x - r_vector[0])**2 + (y - r_vector[1])**2 + (z - r_vector[2])**2 - radius**2
        osculating_plane = self.osculating_plane(p)
        return osculating_sphere, osculating_plane

    # ======================================= Helper methods ======================================= #
    @staticmethod
    def _find_plane(p: tuple, vector1: np.ndarray, vector2: np.ndarray) -> Expr:
        """ Helper method tp find plane that fits point p and two non-collinear vectors """
        x, y, z = symbols('x, y, z')
        xyz = np.array([x, y, z])
        matrix = Matrix([xyz - p, vector1, vector2])
        plane = matrix.det()
        assert plane != Zero, 'vector1 and vector2 must be non-collinear'
        return plane


if __name__ == '__main__':
    # time without caching = 0.4286801815032959
    # time with caching = 0.14731216430664062
    # 2.9 times faster

    # test
    point = (1, 1, 1)
    curve = GeneralCurve3D('x**2+y**2+z**2-3', 'x**2+y**2-2')
    start = time()
    print('For curve {} at point {}:'.format(curve, point))
    print('\ttangent unit vector: {}'.format(curve.tangent_vector(point)))
    print('\tnormal unit vector: {}'.format(curve.normal_vector(point)))
    print('\tbinormal unit vector: {}'.format(curve.binormal_vector(point)))
    print('\tosculating plane: {} = 0'.format(curve.osculating_plane(point)))
    print('\tnormal plane: {} = 0'.format(curve.normal_plane(point)))
    print('\treference plane: {} = 0'.format(curve.reference_plane(point)))
    print('\tcurvature: {}'.format(curve.curvature(point)))
    osc_sphere, osc_plane = curve.osculating_circle(point)
    print('\t# osculating circle <=> intersection of osculating sphere and osculating plane')
    print('\tosculating sphere: {} = 0'.format(osc_sphere))
    print('\tosculating plane: {} = 0'.format(osc_plane))
    print('\nTime elapsed:', time() - start)
