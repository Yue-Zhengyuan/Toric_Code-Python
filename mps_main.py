# 
#   main_mps.py
#   Toric_Code-Python
#   apply the Trotter gates to MPS
#
#   created on Feb 18, 2019 by Yue Zhengyuan
#

import numpy as np
import gates
import para_dict as p
import lattice as lat
import mps
import gnd_state
from str_create import str_create,str_create2,selectRegion,convertToStrOp
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

# create string list (can handle both x-PBC and OBC)
str_list = str_create2(args, args['ny'] - 1)

# save parameters
result_dir = "result_mps_2019_4_25_19_37/"
psi = np.load(result_dir + 'psi.npy')
psi = list(psi)

result_dir += "adiab/"
os.makedirs(result_dir, exist_ok=True)
parafile = result_dir + 'parameters.txt'
resultfile = result_dir + 'undressed_result.txt'
with open(parafile, 'w+') as file:
    file.write(json.dumps(p.args))  # use json.loads to do the reverse
with open(resultfile, 'w+') as file:
    pass

# apply undressed string
# <psi| exp(+iHt) S exp(-iHt) |psi>
for string in tqdm(str_list):
    bond_on_str, area, circum = convertToStrOp(string, args)
    bond_list = [lat.lat(bond_on_str[i][0:2], bond_on_str[i][2], (args['nx'], args['ny']), args['xperiodic']) for i in range(len(bond_on_str))]
    # create string operator
    str_op = []
    for i in range(args['real_n']):
        str_op.append(np.reshape(p.iden, (1,2,2,1)))
    for i in bond_list:
        str_op[i] = np.reshape(p.sx, (1,2,2,1))
    result = mps.matElem(psi, str_op, psi)
    with open(parafile, 'a+') as file:
        file.write('\n')
        file.write(str(area) + '\t' + str(circum) + '\n' + str(bond_on_str) + '\n' + str(bond_list) + '\n')    # bonds
    with open(resultfile, 'a+') as file:
        file.write(str(area) + '\t' + str(circum) + "\t" + str(result) + '\n')
    