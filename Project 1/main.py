# libraries
import sympy as sym
import matplotlib.pyplot as plt
import math
from sympy import *
from sympy.plotting.plot import MatplotlibBackend
import scipy as sp
from scipy.optimize import minimize_scalar

# universal variable x
x = Symbol('x')

# method to read in file data
def read_entries(file_dir):
    file_object = open(file_dir, "r")
    lines = file_object.readlines()
    file_object.close()
    return lines


# method to process data that was read from the file
def process_entries(lines):
    func = sym.simplify(lines[0])
    clean_interval_str = lines[1].split(',')
    endpoint_1 = float(clean_interval_str[0].replace("[", ''))
    endpoint_2 = float(clean_interval_str[1].replace("]", ''))
    interval = Interval(endpoint_1, endpoint_2)
    appx_pnt = float(lines[2])
    max_error = float(lines[3])
    mid_point = (endpoint_1 + endpoint_2) / 2
    return func, interval, appx_pnt, max_error, mid_point, endpoint_1, endpoint_2


# method to find the local maximum of a function
def find_max(func, left_end, right_end):
    neg_func = sym.lambdify(x, -func)  # negate the function since minimize_scalar only finds minimums
    max = -minimize_scalar(neg_func, bounds=(left_end, right_end)).fun  # return the negate of local minimum
    print(max)
    return max


# function to find the degree of first n-th degree polynomial that satisfies the max error
def find_first_deg(func, left_end, right_end, center, max_error):
    n = 0
    func = func.diff(x)  # f_(1)
    max = find_max(Abs(func), left_end, right_end)  # maximum of absolute value of f_(1)
    bound = (((abs(x - center) ** (n + 1)) / factorial(n + 1)) * max).subs(x, right_end)  # error bound at f_(1)
    while bound > max_error:
        n = n + 1
        func = func.diff(x)  # f_(n+1) with n > 0
        max = find_max(Abs(func), left_end, right_end)  # maximum of absolute value of f_(n+1)
        bound = (((abs(x - center) ** (n + 1)) / factorial(n + 1)) * max).subs(x, right_end)  # error bound at f_(n+1)
    return n  # first n that satisfies error bound <= max_error


# function to print out the p_n(x) that satisfies the max error
def print_first_deg(func, degree, center):
    nth_poly = ''
    center_str = Symbol(str(center))
    # loop for summation
    for k in range(degree + 1):
        nth_poly = nth_poly + str(sym.simplify(func.subs(x, center) * ((x - center) ** k)) / factorial(k)) + " +\n "
        func = diff(func, x)
    return nth_poly[:-3]


# function to evaluate p_n(c)
def eval_at_c(func, degree, center, c):
    return sym.simplify(print_first_deg(func, degree, center)).subs(x, c)


# function to plot values
def plot_funcs(func1, func2, left_end, right_end, eval_pnt):
    # func1 is f(x) and func2 is p_n(x)
    func2 = sym.simplify(func2)
    p1 = plot(func1, (x, left_end, right_end), line_color='red', legend=True, show=False)
    p1[0].label = 'f(x)'
    p2 = plot(func2, (x, left_end, right_end), line_color='blue', legend=True, show=False, )
    p2[0].label = 'p_n(x)'
    p1.append(p2[0])
    be = MatplotlibBackend(p1)
    be.process_series()
    plt = be.plt
    # draw (c, f(c))
    plt.plot([eval_pnt], [func1.subs(x, eval_pnt)], 'o', markersize=4, label="(c, f(c)", color='red')
    # draw (c, p_n(c))
    plt.plot([eval_pnt], [func2.subs(x, eval_pnt)], 'o', markersize=4, label="(c, p_n(c)", color='blue')
    plt.legend()
    plt.show()


# Main
if __name__ == '__main__':
    print("Please input your txt file directory:")
    file_dir = input().replace('"', '')
    # read file
    lines = read_entries(file_dir)
    # process file
    entries = process_entries(lines)
    func = entries[0]  # f(x)
    interval = entries[1]  # interval
    appx_pnt = entries[2]  # point to approximate at
    max_error = entries[3]  # max error bound
    mid_point = entries[4]  # mid point of interval
    left_end = entries[5]  # left end of the interval
    right_end = entries[6]  # right end of the interval
    # goal of the program
    print("The function f(x) = \n", pretty(func), "\nwith domain", pretty(interval),
          "will be approximated at the point c =", appx_pnt, " using a Taylor polynomial "
                                                             "centered at", mid_point,
          ". \nThis approximation will be accurate to within an error of E =", max_error, ".")
    first_deg = find_first_deg(func, left_end, right_end, mid_point, max_error) # find n
    # print out the n-th polynomial p_n(x)
    print("The first Taylor's polynomial p_n(x) centered at the midpoint of", pretty(interval), " that is certain to "
          "approximate f(x) \nto within E =",
          max_error, "on all of", pretty(interval), "is with degree n =", first_deg,
          ' of the form \np_n(x) =', print_first_deg(func, first_deg, mid_point), '.')
    f_at_c = func.subs(x, appx_pnt)  # find f(c)
    appx_at_c = eval_at_c(func, first_deg, mid_point, appx_pnt)  # find p_n(c)
    print("f(c) =", f_at_c.round(6), "and p_n(c) =", appx_at_c.round(6))  # round to 6 decimals
    print("R_n(c) = ", (appx_at_c - f_at_c).round(6))  # find R_n(c) and round to 6 decimals
    # plot functions and values
    plot_funcs(func, print_first_deg(func, first_deg, mid_point), left_end, right_end, appx_pnt)
