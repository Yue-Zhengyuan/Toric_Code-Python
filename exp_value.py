# 
#   exp_value.py
#   Toric_Code-Python
#   calculate expectation value of string operator at ground state
#
#   created on Apr 28, 2019 by Yue Zhengyuan
#

import numpy as np
import gates
import para_dict as p
import lattice as lat
import mps
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
# save parameters
str_sep = int(args['ny']/2)
result_dir = "mps_adiab_2019-04-29_15-19/"
resultfile = result_dir + 'undressed_result' + str(str_sep) + '.txt'
with open(resultfile, 'w+') as file:
    pass

for nx in tqdm(range(5,6)):
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
    string = crt.str_create3(args, str_sep)[0]
    psi_name = result_dir + "mps_{}by{}_hz-{:.2f}.npy".format(args['nx'], args['ny'], 0.3)
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
    result = mps.matElem(psi, str_op, psi)
    with open(resultfile, 'a+') as file:
        file.write(str(area) + '\t' + str(circum) + "\t" + str(result) + '\n')
        