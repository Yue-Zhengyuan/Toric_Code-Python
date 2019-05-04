# 
#   apply_string.py
#   Toric_Code-Python
#   apply the evolved string to Toric Code ground state
#
#   created on Feb 18, 2019 by Yue Zhengyuan
#

import numpy as np
import gates, mps, gnd_state
import para_dict as p
import lattice as lat
import str_create as crt
import sys, time, datetime, os, json, ast
from copy import copy
from tqdm import tqdm
from itertools import product

args = copy(p.args)

result_dir = "test/"

for sep in [10]:
    resultfile = result_dir + \
        'dressed_result_sep-{}_hz-{:.2f}.txt'.format(sep, args['hz'])
    with open(resultfile, 'w+') as file:
        pass
    for nx in range(4, 5):
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
        psi = np.load("mps_adiab_2019-04-30_23-26/mps_4by20_hz-0.40.npy")

        # restore information of the string
        closed_str_list = crt.str_create3(args, sep)
        string = closed_str_list[0]
        bond_on_str, area, circum = crt.convertToStrOp(string, args)
        bond_list = [lat.lat(bond_on_str[i][0:2], bond_on_str[i][2], (args['nx'], args['ny']), args['xperiodic']) for i in range(len(bond_on_str))]

        op = np.load(result_dir + 'quasi_op_{}by{}_sep-{}_hz-{:.2f}.npy'.format(args['nx'], args['ny'], sep, args['hz']), allow_pickle=True)
        op = list(op)
        exp_value = mps.matElem(psi, op, psi, verbose=True)
        with open(resultfile, 'a+') as file:
            file.write("{}\t{}\t{}\n".format(area, circum, exp_value))
