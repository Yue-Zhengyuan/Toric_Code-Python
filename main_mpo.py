# 
#   main_mpo.py
#   Toric_Code-Python
#   apply the gates to MPO and monitoring entanglement
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
import sys
import copy
import time
import datetime
import os
import json
import ast
from tqdm import tqdm

args = copy.copy(p.args)
# clear magnetic field
args['hz'] = 0.0

# create result directory
# result_dir = sys.argv[1]
nowtime = datetime.datetime.now()
result_dir = '_'.join(['result_mpo', str(nowtime.year), str(nowtime.month), 
str(nowtime.day), str(nowtime.hour), str(nowtime.minute)])
os.makedirs(result_dir, exist_ok=True)
# save parameters
with open(result_dir + '/parameters.txt', 'w+') as file:
    file.write(json.dumps(p.args))  # use json.loads to do the reverse

# create closed string operator MPO enclosing different area(S)
# read string from file
with open('list.txt', 'r') as f:
    line = f.readlines()
# string_no = int(sys.argv[2])
# string_no = 22
# bond_on_str, str_area = lat.convertToStrOp(ast.literal_eval(line[string_no]))
bond_on_str = [(4,0,'d'),(4,1,'d'),(4,2,'d'),(4,3,'d')]
str_area = len(bond_on_str)
# create string operator
str_op = []
for i in range(args['real_n']):
    str_op.append(np.reshape(p.iden, (1,2,2,1)))
# convert coordinate to unique number in 1D
bond_list = []
for bond in bond_on_str:
    bond_list.append(lat.lat(bond[0:2], bond[2], (args['nx'], args['ny']), args['xperiodic']))
for i in bond_list:
    str_op[i] = np.reshape(p.sx, (1,2,2,1))
with open(result_dir + '/parameters.txt', 'a+') as file:
    file.write('\nUsing String Operators:\n')
    file.write(str(str_area) + '\t' + str(bond_on_str) + '\n')    # bonds

step = 0
virDim_list = list(str_op[i].shape[0] for i in range(len(str_op)))
with open(result_dir + '/dim_monitor.txt', 'w+') as file:
    file.write("Step\tMaximum virtual bond dimension\n")
    file.write(str(step) + '\t' + str(max(virDim_list)) + '\n')
step += 1

# create Toric Code ground state |psi>
# psi = gnd_state.gnd_state_builder(args)

tstart = time.perf_counter()

# adiabatic evolution: 
# exp(-iH't) S exp(+iH't)
stepNum = int(p.args['ttotal']/p.args['tau'])
iterlist = np.linspace(0, p.args['hz'], num = stepNum+1, dtype=float)
iterlist = np.delete(iterlist, 0)
iterlist = np.flip(iterlist)
timestep = args['ttotal']/stepNum
for hz in iterlist:
    args['hz'] = hz
    gateList = gates.makeGateList(str_op, args)
    str_op = mpo.gateTEvol(str_op, gateList, timestep, timestep, args=args)
    virDim_list = list(str_op[i].shape[0] for i in range(len(str_op)))
    with open(result_dir + '/dim_monitor.txt', 'a+') as file:
        file.write(str(step) + '\t' + str(max(virDim_list)) + '\n')
    step += 1

tend = time.perf_counter()
with open(result_dir + '/parameters.txt', 'a+') as file:
    file.write("\nAdiabatic evolution: " + str(tend-tstart) + " s\n")
mpo.save_to_file(str_op, result_dir + '/str_op.txt')