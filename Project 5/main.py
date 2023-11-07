import sympy as sym
from scipy.optimize import minimize_scalar
from sympy import *
import scipy as sp
from prettytable import PrettyTable

x = Symbol('x')


# function top find the smallest n's that fit the error and their max
def find_n(func, start, end, error):
    second_d = diff(diff(func, x), x)  # f''(x)
    fourth_d = diff(diff(second_d), x)  # f^(4)(x)
    max_sec = find_max(second_d, start, end)  # bound of f''(x)
    max_fourth = find_max(fourth_d, start, end)  # bound of f^(4)(x)
    count_trap = 1  # counter for trapezoid's error
    count_simp = 1  # counter for Simpson's error
    # check for smallest n that fit trapezoidal error bound
    while (end - start) ** 3 * max_sec / (12 * (count_trap ** 2)) > error:
        count_trap += 1
    # check for smallest n that fit Simpson's error bound
    h = (end - start) / (2 * (count_simp ** 4))
    while (4 * (h ** 4) * (end - start) * max_fourth / 45) > error:
        count_simp += 1
        h = (end - start) / (2 * (count_simp ** 4))
    counter = max(count_simp, count_trap)
    return counter, count_trap, count_simp, second_d, fourth_d, max_sec, max_fourth


# perform trapezoidal rule for n = 1,2,..., max{M,N}
def trapezoidal(func, n, start, end):
    list_of_appx = []
    # loop through each n
    for order in range(1, n + 1):
        t_n = 0
        # loop to calculate sum of f(x_i)'s, does not calculate f(a) and f(b) separately
        for term in range(order):
            delta = (end - start) / order  # delta = (b-a)/n
            x_i = start + term * delta  # x_i = a + i*delta
            x_i1 = start + (term + 1) * delta  # x_{i+1} = a + (i+1)*delta
            # f(a) and f(b) are only counted once
            t_n += func.subs(x, x_i) + func.subs(x, x_i1)
        t_n = t_n * (end - start) / (2 * order)  # *(b-a)/12
        list_of_appx.append(t_n)
    return list_of_appx


# perform Simpson's rule for n = 1,2,..., max{M,N}
def simpons_rule(func, n, start, end):
    list_of_appx = []
    # loop through each n
    for order in range(1, n + 1):
        s_n = 0
        h = (end - start) / (2 * order)  # h = (b-a)/2n
        # loop to differentiate f of odd and even terms
        for k in range(1, 2 * order):
            if k % 2 == 1:  # if odd, times 4
                s_n += 4 * func.subs(x, start + k * h)
            else:  # if even, times 2
                s_n += 2 * func.subs(x, start + k * h)
        s_n += func.subs(x, start) + func.subs(x, end)  # + f(a) + f(b)
        s_n = s_n * h / 3  # * h/3
        list_of_appx.append(s_n)
    return list_of_appx


# Method to read in file data
def read_entries(file_dir):
    file_object = open(file_dir, "r")
    lines = file_object.readlines()
    file_object.close()
    return lines


# calculate error for each approximation
def calc_error(true_value, trap_values, simp_values):
    trap_error = []
    simp_error = []
    for value in trap_values:
        trap_error.append(true_value - value)
    for value in simp_values:
        simp_error.append(true_value - value)
    return trap_error, simp_error


def find_max(func, left_end, right_end):
    neg_func = sym.lambdify(x, -abs(func))  # negate the error func since minimize_scalar only finds minimums
    max = -minimize_scalar(neg_func, bounds=(left_end, right_end)).fun  # return the negate of local minimum
    return max


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
    error = float(lines[2])
    print("The integral of the function f(x) =", func, "over the interval ", pretty(interval), "will be computed using "
                                                                                               "the fundamental "
                                                                                               "theorem of "
                                                                                               "calculus.\nIt will "
                                                                                               "also be approximated "
                                                                                               "to within an error "
                                                                                               "tolerance of",
          str(error), 'using the trapezoid rule and Simpson\'s rule.')
    # using sympy to find antiderivative
    antd_func = integrate(func, x)
    true_value = antd_func.subs(x, end_point) - antd_func.subs(x, start_point)
    print('An antiderivative of f(x) is F(x) =', antd_func, '. By the fundamental theorem of calculus, the integral '
                                                            'of f(x) over', pretty(interval), '\nis F(', end_point,
          ') - F(', start_point, ') =', round(true_value, 10), '.')
    info = find_n(func, start_point, end_point, error)
    n = info[0]  # max{M,N}
    trapezoidal_appx = trapezoidal(func, n, start_point, end_point)  # trapezoidal's approximations from 1 to n
    simpsons_appx = simpons_rule(func, n, start_point, end_point)  # simpson's approximations from 1 to n
    error_lists = calc_error(true_value, trapezoidal_appx, simpsons_appx)
    trapezoidal_err = error_lists[0]
    simpsons_err = error_lists[1]
    table_1 = PrettyTable()
    table_1.field_names = ['', 'Trapezoidal', 'Simpson\'s']
    table_1.add_row(['Derivative', 'f\'\'(x)= ' + str(info[3]), 'f^(4)(x) = ' + str(info[4])])
    table_1.add_row(['Bound on ' + pretty(interval), round(info[5], 10), round(info[6], 10)])
    table_1.add_row(['n =', str(info[1]) + ' trapezoids', str(info[2]) + ' subintervals'])
    print(table_1)
    table_2 = PrettyTable()
    table_2.field_names = ["n", 'Trapezoidal', 'Simpson\'s', 'Trapezoid Error', 'Simpson\'s Error']
    for i in range(n):
        table_2.add_row(
            [i + 1, round(trapezoidal_appx[i], 10), round(simpsons_appx[i], 10), round(trapezoidal_err[i], 10),
             round(simpsons_err[i], 10)])
    print(table_2)
