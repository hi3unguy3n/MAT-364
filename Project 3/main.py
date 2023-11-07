# libraries
import math
import sympy as sym
from sympy import *
import scipy as sp
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from sympy.plotting.plot import MatplotlibBackend

x = Symbol('x')
a = Symbol('a')


# Method to read in file data
def read_entries(file_dir):
    file_object = open(file_dir, "r")
    lines = file_object.readlines()
    file_object.close()
    return lines


# Bisection method
def bisection(func, a, b, error):
    m = float((a + b) / 2)  # m_0
    count = 1 # counter
    while b - m > error:
        if sign(func.subs(x, a)) * sign(func.subs(x, m)) > 0:    # if sign(f(a))*sign(f(m)) > 0
            a = m
        else: # if sign(f(a))*sign(f(m)) < 0
            b = m
        m = float((a + b) / 2)
        count = count + 1
    return m, count


# Newton's method
def newtons(func, a, b, error):
    x_n = (a + b) / 2  # x_0
    f_prime = diff(func, x)  # f'(x)
    x_n1 = x_n - (func.subs(x, x_n) / f_prime.subs(x, x_n))  # x_1 = x_n - f(x_n)/f'(x_n)
    count = 1  # counter
    while Abs(x_n1 - x_n) > error:  # if |x_(n+1) - x_n| > epsilon error
        x_n = x_n1
        x_n1 = x_n - (func.subs(x, x_n) / f_prime.subs(x, x_n))
        count = count + 1
    return x_n1, count


# find interval of length one containing root this is a modified Newton's method to find such interval. Starting from
# x_n = 0, we check sign(f(x_n))*sign(f(x_n-1)) and sign(f(x_n))*sign(f(x_n+1)). If one of the signs is less than or
# equal to 0, the interval must be [x_n, x_n-1] or [x_n, x_n+1] (endpoints must be properly ordered). If not,
# Newton's method is applied to find the new x_n that converges to the real root whose interval [x_n, x_n-1] or
# [x_n, x_n+1] contains the root.
def find_interval(func):
    x_n = 0  # x_0 = 0
    c = 0   # placeholder for x_n - 1 or x_n + 1
    if sign(func.subs(x, x_n)) * sign(func.subs(x, x_n + 1)) <= 0:  # check sign(f(x_0))*sign(f(x_0+1))
        c = 1
    if sign(func.subs(x, x_n)) * sign(func.subs(x, x_n - 1)) <= 0:  # check sign(f(x_0))*sign(f(x_0-1))
        c = -1
    f_prime = diff(func, x)   # f'(x)
    # Apply Newton's method if both signs are not negative
    while sign(func.subs(x, x_n)) * sign(func.subs(x, x_n + 1)) > 0 and sign(func.subs(x, x_n)) * sign(
            func.subs(x, x_n - 1)) > 0:
        x_n1 = x_n - (func.subs(x, x_n) / f_prime.subs(x, x_n))
        x_n = x_n1
    if sign(func.subs(x, x_n)) * sign(func.subs(x, x_n + 1)) <= 0:  # check sign(f(x_n))*sign(f(x_n+1))
        c = x_n + 1
    if sign(func.subs(x, x_n)) * sign(func.subs(x, x_n - 1)) <= 0:  # check sign(f(x_n))*sign(f(x_n-1))
        c = x_n - 1
    return sorted([x_n, c])  # return sorted endpoints


# Secant method
def secant(func, a, b, error):
    x_n_minus_1 = a  # x_0
    x_n = b  # x_1
    x_n_plus_1 = x_n - ((x_n - x_n_minus_1) / (func.subs(x, x_n) - func.subs(x, x_n_minus_1))) * func.subs(x,
                                                                                                           x_n)  # x_2
    count = 1
    while Abs(x_n_plus_1 - x_n) > error:
        x_n_minus_1 = x_n
        x_n = x_n_plus_1
        x_n_plus_1 = x_n - ((x_n - x_n_minus_1) / (func.subs(x, x_n) - func.subs(x, x_n_minus_1))) * func.subs(x, x_n)
        count = count + 1
    return x_n_plus_1, count


# Factoring method
def factoring(func, error):
    interval = find_interval(func)
    start = interval[0]
    end = interval[1]
    # root-finding using the bisection method
    root = bisection(func, start, end, error)[0]
    # list of coefficients of polynomial to be factored
    coeff_list = Poly(func).all_coeffs()
    # list of coefficients of the remainder polynomial
    new_coeff_list = [0] * len(coeff_list)
    # b_n = a_n
    new_coeff_list[0] = coeff_list[0]
    for i in range(1, len(coeff_list) - 1):
        # b_(n-1) = a_(n-1) + b_n * root
        new_coeff_list[i] = coeff_list[i] + new_coeff_list[i - 1] * root
    # remove b_0
    new_coeff_list.pop(-1)
    return root, new_coeff_list, start, end

# Plotting function
def plot_funcs(func1, func2, left_end, right_end):
    func2 = sym.simplify(func2)
    p1 = plot(func1, (x, left_end, right_end), line_color='black', legend=True, show=False)
    p1[0].label = 'p(x)'
    p2 = plot(func2, (x, left_end, right_end), line_color='yellow', legend=True, show=False)
    p2[0].label = 'factorization'
    p1.append(p2[0])
    p1.show()


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
    interval = Interval(start_point, end_point)
    # read error
    err = float(lines[2])
    decimal_places = len(lines[2].split(".")[1])  # decimal places of error plus 1
    # read_p(x)
    poly_func = sym.simplify(lines[3])
    print("The root of function f(x) =\n", pretty(func), "\nwithin", pretty(interval), "will be approximated within an "
                                                                                       "error tolerance of ", err,
          " using the bisection method, Newton's method, and the secant method.")
    bisection_method = bisection(func, start_point, end_point, err)
    newtons_method = newtons(func, start_point, end_point, err)
    secant_method = secant(func, start_point, end_point, err)
    table = PrettyTable()
    table.field_names = ['Method', 'Appx_x', 'Count']
    table.add_row(['Bisection', round(bisection_method[0], decimal_places), bisection_method[1]])
    table.add_row(['Newtons', round(newtons_method[0], decimal_places), newtons_method[1]])
    table.add_row(['Secant', round(secant_method[0], decimal_places), secant_method[1]])
    print(table)
    order = 0  # counter i for p[i](x)
    print("The polynomial that is to be factored is p[" + str(order) + "](x) =", poly_func, "\n")
    root_list = []  # list of roots after done factoring
    new_poly = poly_func  # placeholder for polynomials
    while degree(new_poly) != 0:  # check if p[i](x) is not constant, if it is then we're done
        factor_info = factoring(new_poly, err)  # factoring method
        root_list.append(factor_info[0])  # add root to list
        start = float(factor_info[2])   # first endpoint of length one interval
        end = float(factor_info[3])   # second endpoint of length one interval
        print("Since p[" + str(order) + "](x)(" + str(start) + ") =", new_poly.subs(x, start),
              " and p[" + str(order) + "](x)(" + str(end) + ") =", new_poly.subs(x, end),
              ", p[" + str(order) + "](x) must have a root in the interval\n[" + str(start) + "," + str(
                  end) + "]. The bisection method finds that root to be approximately " + str(
                  factor_info[0]) + ". Factoring the corresponding linear term out of p[" + str(
                  order) + "](x) \nproduces p[" + str(order + 1) + "](x) =",
              simplify(Poly(factor_info[1], x).subs(a, 1)), ".\n")
        new_poly = Poly(factor_info[1], x)  # p[i+1](x)
        order = order + 1
    constant = round(new_poly.subs(a, 0), decimal_places)   # constant after done factoring
    # print roots
    print("The root(s) of p(x) is/are:")
    for i in range(len(root_list)):
        print("x_", str(i), "=", str(root_list[i]))
    # print factorization
    print("The approximate factorization of p(x) is ")
    factorization = ""
    if constant != 1:
        factorization = factorization + str(constant) + "*"
    for root in root_list:
        if root > 0:
            factorization = factorization + "(x-" + str(round(root, decimal_places)) + ")*"
        else:
            factorization = factorization + "(x+" + str(round(Abs(root), decimal_places)) + ")*"
    factorization = factorization.rstrip(factorization[-1])
    print(factorization)
    # plot functions
    plot_funcs(poly_func, factorization, -1000, 1000)
