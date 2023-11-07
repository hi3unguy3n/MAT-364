# libraries
import sympy as sym
import math
from sympy import *
import scipy as sp
from prettytable import PrettyTable

n = Symbol('n')
k = Symbol('k')


# method to read in file data
def read_entries(file_dir):
    file_object = open(file_dir, "r")
    lines = file_object.readlines()
    file_object.close()
    return lines


def round_sig(x, n):
    y = 0
    if x != 0:
        e = int(math.floor(math.log10(abs(x))))
        y = round(x * 10 ** (n - e - 1)) * 10 ** (e - n + 1)
    return y


def truncate_sig(x, n):
    y = 0
    if x != 0:
        e = int(math.floor(math.log10(abs(x))))
        y = int(x * 10 ** (n - e - 1)) * 10 ** (e - n + 1)
    return y


def calc_coefs(coef, n_value):
    list = []
    for index in range(1, n_value + 1):
        kth_coef = coef.subs(k, index)
        list.append(float(kth_coef))
    return list


def sort_sl(list):
    list.sort(reverse=False, key=Float)
    return list


def sum_ls_with_chop(values):
    sum = 0
    for value in reversed(values):
        sum = truncate_sig(sum + value, 10)
    return sum


def sum_ls_with_round(values):
    sum = 0
    for value in reversed(values):
        sum = round_sig(sum + value, 10)
    return sum


def sum_sl_with_chop(values):
    sum = 0
    for value in values:
        sum = truncate_sig(sum + value, 10)
    return sum


def sum_sl_with_round(values):
    sum = 0
    for value in values:
        sum = round_sig(sum + value, 10)
    return sum


def create_table_entries(func, kth_coef, list_of_trials):
    entries = []
    for trial in list_of_trials:
        f_value = round_sig(func.subs(n, trial), 10)
        coef_list = calc_coefs(kth_coef, trial)
        sorted_list = sort_sl(coef_list)
        ls_chop_err = round_sig(f_value - sum_ls_with_chop(sorted_list), 10)
        ls_round_err = round_sig(f_value - sum_ls_with_round(sorted_list), 10)
        sl_chop_err = round_sig(f_value - sum_sl_with_chop(sorted_list), 10)
        sl_round_err = round_sig(f_value - sum_sl_with_round(sorted_list), 10)
        entries.append(
            dict({'n': trial, 'f_n': f_value, 'E_1': ls_chop_err, 'E_2': ls_round_err, 'E_3': sl_chop_err,
                  'E_4': sl_round_err}))
    return entries


def draw_table(entries):
    table = PrettyTable()
    table.field_names = ['n', 'f(N_i)', 'ls_chop', 'ls_round', 'sl_chop', 'sl_round']
    for index in range(len(entries)):
        table.add_row(
            [entries[index].get('n'), round_sig(entries[index].get('f_n'), 10),
             round_sig(entries[index].get('E_1'), 10), round_sig(entries[index].get('E_2'), 10),
             round_sig(entries[index].get('E_3'), 10), round_sig(entries[index].get('E_4'), 10)])
    return table


if __name__ == '__main__':
    print("Please input your txt file directory:")
    file_dir = input().replace('"', '')
    # read file
    lines = read_entries(file_dir)
    func = sym.simplify(lines[0])
    kth_coef = sym.simplify(lines[1])
    list_of_trials = lines[2].strip('\n').strip(' ').split(',')
    list_of_trials = [eval(i) for i in list_of_trials]  # turn string to int
    table_info = create_table_entries(func, kth_coef, list_of_trials)
    print("The function f(n) =")
    pprint(func, use_unicode=True)
    print("which equals the finite series ", kth_coef,
          "as k ranges from 1 to n\nwill be approximated at the values in", list_of_trials,
          "by summing terms from smallest-to-largest (SL) and largest-to-smallest (LS).\n"
          "Each method will be done using both chopping and rounding.")
    print(draw_table(table_info))
