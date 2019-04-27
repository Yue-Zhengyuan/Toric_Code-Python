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
psi = gnd_state.gnd_state_builder(args)
# save parameters
result_dir = "test"
os.makedirs(result_dir, exist_ok=True)
with open(result_dir + '/parameters.txt', 'w+') as file:
    file.write(json.dumps(args))  # use json.loads to do the reverse
with open(result_dir + '/entg_entropy.txt', 'w+') as file:
    pass

# psi = np.load('result_mps_2019_4_24_7_28/psi.npy')
# psi = list(psi)

for site in tqdm(range(len(psi))):
    psi, entg = mps.position(psi, site, args, oldcenter=site - 1, compute_entg=True)
    with open(result_dir + '/entg_entropy.txt', 'a+') as file:
        file.write(str(site) + '\t' + str(entg) + '\n')