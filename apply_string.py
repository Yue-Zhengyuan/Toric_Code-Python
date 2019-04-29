# 
#   apply_string.py
#   Toric_Code-Python
#   apply the evolved string to Toric Code ground state
#
#   created on Feb 18, 2019 by Yue Zhengyuan
#

import numpy as np
import gates
import para_dict as p
import lattice as lat
import mps
import mpo
import gnd_state
import str_create as crt
import sys
from copy import copy
import time
import datetime
import os
import json
import ast
from tqdm import tqdm

args = copy(p.args)
# clear magnetic field
args['hz'] = 0.0

result_dir = "result_mpo_pair/"

for str_sep in [6, 10, 14]:
    exp_value_file = result_dir + 'undressed_result_' + str(str_sep) + '.txt'
    with open(exp_value_file, 'w+') as file:
        pass

for nx in range(3, 8): 
    args['nx'] = nx
    n = 2 * (args['nx'] - 1) * args['ny']
    # Y-non-periodic
    n -= args['nx'] - 1
    # X-non-periodic
    n += args['ny'] - 1
    args['n'] = n
    # n in case of periodic X
    if args['xperiodic'] == True:
        args['real_n'] = n - (args['ny'] - 1)
    else:
        args['real_n'] = n
    # create Toric Code ground state
    psi = gnd_state.gnd_state_builder(args)

    # restore information of the string
    closed_str_list = crt.str_create3(args, str_sep)
    string = closed_str_list[0]
    bond_on_str, area, circum = crt.convertToStrOp(string, args)
    bond_list = [lat.lat(bond_on_str[i][0:2], bond_on_str[i][2], (args['nx'], args['ny']), args['xperiodic']) for i in range(len(bond_on_str))]

    for str_sep in [6, 10, 14]:
        op = np.load(result_dir + 'adiab_op_' + str(nx) + 'by20_' + str(str_sep) + '.npy')
        op = list(op)
        exp_value = mps.matElem(psi, op, psi)
        exp_value_file = result_dir + 'undressed_result_' + str(str_sep) + '.txt'
        with open(exp_value_file, 'a+') as file:
            file.write(str(area) + '\t' + str(circum) + '\t' + str(exp_value) + '\n')
