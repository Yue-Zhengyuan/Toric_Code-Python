# 
#   mps_quasi_evol.py
#   Toric_Code-Python
#   inverse quasi adiabatic evolution based on adiabatic results
#
#   created on Feb 18, 2019 by Yue Zhengyuan
#

import numpy as np
import gates
import para_dict as p
import lattice as lat
import mps
import gnd_state
from str_create import str_create, str_create2, selectRegion, convertToStrOp
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

# create Toric Code ground state |psi>
result_dir = sys.argv[1]
# result_dir = "result_mps_2019_4_24_11_37"

psi_dir = os.path.split(result_dir)[0]
psi = np.load(psi_dir + "/psi.npy")
psi = list(psi)

resultfile = result_dir + '/dressed_result.txt'
with open(resultfile, 'w+') as file:
    pass

# create string list (can handle both x-PBC and OBC)
str_list = str_create(args, args['ny'] - 1)

# select string
str_no = int(sys.argv[2])
# str_no = -1
string = str_list[str_no]
bond_on_str, area, circum = convertToStrOp(string, args)
bond_list = [lat.lat(bond_on_str[i][0:2], bond_on_str[i][2], (args['nx'], args['ny']), args['xperiodic']) for i in range(len(bond_on_str))]
bond_list.sort()
region = selectRegion(bond_on_str, 1, args)

# create string operator
str_op = []
for i in range(args['real_n']):
    str_op.append(np.reshape(p.iden, (1,2,2,1)))
for i in bond_list:
    str_op[i] = np.reshape(p.sx, (1,2,2,1))

# quasi-adiabatic evolution: exp(-iHt)|psi> (With field along z)
print("Quasi-Adiabatic Evolution")
tstart = time.perf_counter()
stepNum = int(p.args['ttotal']/p.args['tau'])
iterlist = np.linspace(0, p.args['hz'], num = stepNum+1, dtype=float)
iterlist = np.delete(iterlist, 0)
# reverse order: decreasing magnetic field
iterlist = np.flip(iterlist)
timestep = args['ttotal']/stepNum
args['g'] *= -1
for hz in tqdm(iterlist):
# for hz in iterlist:
    args['hz'] = -hz
    # only use part of the Hamiltonian within some distance to the string
    gateList = gates.makeGateList(args['real_n'], args, region=region)
    psi = mps.gateTEvol(psi, gateList, timestep, args['tau'], args=args)
tend = time.perf_counter()

result = mps.matElem(psi, str_op, psi)

with open(resultfile, 'a+') as file:
    file.write(str(area) + '\t' + str(circum) + "\t" + str(result) + '\n')
    