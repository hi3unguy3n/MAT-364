import sympy as sym
from scipy.optimize import minimize_scalar
from sympy import *
import scipy as sp
from prettytable import PrettyTable
from sympy.core.rules import Transform


x = Symbol('x')


# Method to read in file data
def read_entries(file_dir):
    file_object = open(file_dir, "r")
    lines = file_object.readlines()
    file_object.close()
    return lines


# Taylor polynomial approximation
def taylors(func, degree, center):
    nth_poly = ''
    # loop for summation
    for k in range(degree + 1):
        nth_poly = nth_poly + str((func.subs(x, center) * ((x - center) ** k)) / factorial(k)) + " +\n "
        func = diff(func, x)
    # round coefficients to 6 decimals
    return simplify(nth_poly[:-3]).xreplace(Transform(lambda x: x.round(6), lambda x: isinstance(x, Float)))


# Interpolation using Lagrange basis functions
def interpolation(func, degree, start, end):
    delta = (end - start) / (degree + 2)
    points = []  # n+1 equally spaced points to approximate
    y = []  # y_i = f(x_i) for i = 0,1,...,n
    for i in range(1, degree + 2, 1):  # note that i = 1,2,...,n+1
        points.append(start + i * delta)
    for index in range(len(points)):  # calculate f(x_i)
        y.append(func.subs(x, points[index]))
    # using Lagrange basis functions to calculate the polynomial and round coefficients to 6 decimals
    q_n = expand(lagrange_interpolation(points, y)).xreplace(
        Transform(lambda x: x.round(6), lambda x: isinstance(x, Float)))
    return q_n, points


# Near-minimax approximation
def near_minimax(func, degree, start, end):
    t = ((start + end) + x * (end - start)) / 2  # change of variable to turn domain to [-1,1]
    new_func = func.subs(x, t)  # sub in new variable
    nodes = []  # list of Chebyshev nodes
    y = []  # y_i = f(x_i) where x_i are the nodes
    chebyshev_formula = sym.simplify(cos((2 * x + 1) * pi / (2 * degree + 2)))
    for i in range(degree + 1):
        nodes.append(float(chebyshev_formula.subs(x, i)))  # calculate Chebyshev nodes
    for i in range(len(nodes)):
        y.append(new_func.subs(x, nodes[i]))  # calculate y_i
    modified = lagrange_interpolation(nodes, y)  # interpolation using Chebyshev nodes
    x_0 = (2 * x - (start + end)) / (end - start)  # change of variable to return to original domain
    # sub the new variable in and round the coefficients to 6 decimals
    near_minimax = expand(modified.subs(x, x_0).evalf()).xreplace(
        Transform(lambda x: x.round(6), lambda x: isinstance(x, Float)))
    return near_minimax, nodes


# Interpolation using Lagrange basis functions
def lagrange_interpolation(points, values):
    L = [1] * len(points)  # list of dummy values for basis functions
    for index in range(len(points)):
        # check index and loop through points at other indices
        for j in range(len(points)):
            if j != index:
                L[index] = sym.simplify(L[index] * (x - points[j]) / (points[index] - points[j]))
    poly = 0  # dummy poly
    for i in range(len(values)):
        poly = poly + simplify(L[i] * values[i])  # poly = sum(L_i * x_i), i = 0,1,...,n
    return poly


# find uniform norm of the error f(x) - p(x)
def find_max(f_x, list_of_funcs, left_end, right_end):
    list_of_max = []
    for func in list_of_funcs:
        neg_func = sym.lambdify(x, -abs(f_x - func))  # negate the error func since minimize_scalar only finds minimums
        max = -minimize_scalar(neg_func, bounds=(left_end, right_end)).fun  # return the negate of local minimum
        list_of_max.append(max)
    return list_of_max


if __name__ == '__main__':
    print("Please input your txt file directory:")
    file_dir = input().replace('"', '')
    # read file
    lines = read_entries(file_dir)
    # read f(x)
    func = sym.simplify(lines[0])
    # read interval
    clean_interval_str = lines[1].split(',')
    start_point = float(clean_interval_str[0].replace("[", ''))
    end_point = float(clean_interval_str[1].replace("]", ''))
    mid_point = (start_point + end_point) / 2
    interval = Interval(start_point, end_point)
    degree = int(lines[2])
    # calculate approximation functions
    p_n = taylors(func, degree, mid_point)
    q_n = interpolation(func, degree, start_point, end_point)
    r_n = near_minimax(func, degree, start_point, end_point)
    print("The function f(x) =\n", pretty(func), "\nwill be approximated on ", pretty(interval), "by polynomials of "
                                                                                                 "degree n =",
          degree, "including the Taylor Polynomial p_", degree, "(x) centered at", mid_point, ",\nthe polynomial q_",
          degree, "(x) that interpolates f(x) at", degree + 1, "equally spaced points from", q_n[1][0], "to", q_n[1][-1]
          , ", and the near-minimax polynomial r_", degree, "(x).")
    print('-' * 130)
    # print out Chebyshev nodes rounded to 6 decimals
    print("The", degree + 1, "Chebyshev nodes are: ")
    for index, node in enumerate(r_n[1]):
        print("x_", index, "=", round(node, 6))
    print('-' * 130)
    # print approximation functions
    print("p_", degree, "(x) =", p_n)
    print("q_", degree, "(x) =", q_n[0])
    print("r_", degree, "(x) =", r_n[0])
    print('-' * 130)
    # create table
    list_of_funcs = [p_n, q_n[0], r_n[0]]
    list_of_max = find_max(func, list_of_funcs, start_point, end_point)
    table = PrettyTable()
    table.field_names = ['Method', 'Uniform Error']
    table.add_row(['Taylor polynomial', round(list_of_max[0], 10)])
    table.add_row(['Interpolation', round(list_of_max[1], 10)])
    table.add_row(['Near-minimax', round(list_of_max[2], 10)])
    print(table)
