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
import mpo
import gnd_state
from str_create import str_create,str_create2
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

result_dir = "test_hz_2/"
op = np.load(result_dir + 'adiab_op_6by20_10.npy')
op = list(op)
entg_file = result_dir + '/entg_6by20_10.txt'
with open(entg_file, 'w+') as file:
    pass

for site in tqdm(range(len(op))):
    op, entg = mpo.position(op, site, args, oldcenter=site - 1, compute_entg=True)
    with open(entg_file, 'a+') as file:
        file.write(str(site) + '\t' + str(entg) + '\n')
