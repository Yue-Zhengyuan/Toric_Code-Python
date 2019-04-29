# 
#   mpo_quasi_evol.py
#   Toric_Code-Python
#   quasi-adiabatic evolution of the string operator MPO
#
#   created on Apr 29, 2019 by Yue Zhengyuan
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
import copy
import time
import datetime
import os
import json
import ast
from tqdm import tqdm

args = copy.copy(p.args)
hz_max = args['hz']

# create result directory
# result_dir = sys.argv[1]
result_dir = "test_hz_" + str(hz_max) + "/"
# nowtime = datetime.datetime.now()
# result_dir = '_'.join(['result_mpo', str(nowtime.year), str(nowtime.month), 
# str(nowtime.day), str(nowtime.hour), str(nowtime.minute)])
os.makedirs(result_dir, exist_ok=True)
# save parameters
with open(result_dir + '/parameters.txt', 'w+') as file:
    file.write(json.dumps(p.args) + '\n')  # use json.loads to do the reverse

# create closed string operator MPO enclosing different area(S)
str_sep = int(args['ny']/2)
# str_sep = int(sys.argv[3])
closed_str_list = crt.str_create3(args, str_sep)
string = closed_str_list[0]
bond_on_str, area, circum = crt.convertToStrOp(string, args)
bond_list = [lat.lat(bond_on_str[i][0:2], bond_on_str[i][2], (args['nx'], args['ny']), args['xperiodic']) for i in range(len(bond_on_str))]
region = crt.selectRegion(bond_on_str, 1, args)

# create string operator
str_op = []
for i in range(args['real_n']):
    str_op.append(np.reshape(p.iden, (1,2,2,1)))
for i in bond_list:
    str_op[i] = np.reshape(p.sx, (1,2,2,1))

# save parameters
with open(result_dir + '/parameters.txt', 'a+') as file:
    file.write(json.dumps(args) + '\n')  # use json.loads to do the reverse
    # Output information
    file.write(str(area) + '\t' + str(circum) + '\n')
    file.write("bond on str: \n" + str(bond_on_str) + '\n')
    file.write("bond number: \n" + str(bond_list) + '\n')
    file.write("bond within distance 1: \n" + str(region) + '\n\n')

tstart = time.perf_counter()
# quasi-adiabatic evolution (using Hamiltonian in the selected region): 
# S' = exp(-iH't) S exp(+iH't) - hz increasing
stepNum = int(p.args['ttotal']/p.args['tau'])
iterlist = np.linspace(0, p.args['hz'], num = stepNum+1, dtype=float)
iterlist = np.delete(iterlist, 0)
timestep = args['ttotal']/stepNum
args['g'] *= -1
for hz in tqdm(iterlist):
    args['hz'] = -hz
    gateList = gates.makeGateList(len(str_op), args, region=region)
    str_op = mpo.gateTEvol(str_op, gateList, timestep, timestep, args=args)
tend = time.perf_counter()
# save string operator
op_save = np.asarray(str_op)
np.save(result_dir + "/quasi_op.npy", op_save)

# adiabatic evolution (Heisenberg picture): 
# exp(+iH't) S exp(-iH't) - hz decreasing
stepNum = int(p.args['ttotal']/p.args['tau'])
iterlist = np.linspace(0, p.args['hz'], num = stepNum+1, dtype=float)
iterlist = np.delete(iterlist, 0)
iterlist = np.flip(iterlist)
timestep = args['ttotal']/stepNum
for hz in tqdm(iterlist):
    args['hz'] = hz
    gateList = gates.makeGateList(len(str_op), args)
    str_op = mpo.gateTEvol(str_op, gateList, timestep, timestep, args=args)
tend = time.perf_counter()
# save string operator
op_save = np.asarray(str_op)
np.save(result_dir + "/final_op.npy", op_save)
