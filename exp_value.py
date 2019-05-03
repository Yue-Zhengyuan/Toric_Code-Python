# 
#   exp_value.py
#   Toric_Code-Python
#   calculate expectation value of string operator at ground state
#
#   created on Apr 28, 2019 by Yue Zhengyuan
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

for sep, hz in product([6,10,14], np.linspace(0.05, 0.2,num=4, endpoint=True)):
    result_dir = "mps_adiab_2019-04-29_15-54/"
    resultfile = result_dir + \
        'undressed_result_sep-{}_hz-{:.2f}.txt'.format(sep, hz)
    with open(resultfile, 'w+') as file:
        pass
    for nx in range(3,8):
        args['nx'] = nx
        n = 2 * (args['nx'] - 1) * args['ny']
        # Y-non-periodic
        n -= args['nx'] - 1
        # X-non-periodic
        n += args['ny'] - 1
        args['n'] =  n
        # n in case of periodic X
        if args['xperiodic'] == True:
            args['real_n'] = n - (args['ny'] - 1)
        else:
            args['real_n'] = n

        # create string list (can handle both x-PBC and OBC)
        string = crt.str_create3(args, sep)[0]
        psi_name = result_dir + "mps_{}by{}_hz-{:.2f}.npy".format(args['nx'], args['ny'], hz)
        psi = np.load(psi_name)
        psi = list(psi)

        # apply undressed string
        # <psi| exp(+iHt) S exp(-iHt) |psi>
        bond_on_str, area, circum = crt.convertToStrOp(string, args)
        bond_list = [lat.lat(bond_on_str[i][0:2], bond_on_str[i][2], (args['nx'], args['ny']), args['xperiodic']) for i in range(len(bond_on_str))]
        # create string operator
        str_op = []
        for i in range(args['real_n']):
            str_op.append(np.reshape(p.iden, (1,2,2,1)))
        for i in bond_list:
            str_op[i] = np.reshape(p.sx, (1,2,2,1))
        result = mps.matElem(psi, str_op, psi, verbose=True)
        with open(resultfile, 'a+') as file:
            file.write(str(area) + '\t' + str(circum) + "\t" + str(result) + '\n')
        