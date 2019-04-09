# 
#   main_mps.py
#   Toric_Code-Python
#   apply the gates and MPO to MPS
#
#   created on Feb 18, 2019 by Yue Zhengyuan
#

import numpy as np
import gates
import para_dict as p
import lattice as lat
import mps
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
# for OBC
if (args['xperiodic'] != False):
    sys.exit("Wrong BC.")

# create closed string operator MPO enclosing different area(S)
# read string from file
with open('newlist.txt', 'r') as f:
    line = f.readlines()
    
# create Toric Code ground state |psi>
psi = gnd_state.gnd_state_builder(args)

tstart = time.perf_counter()

# adiabatic evolution: exp(-iHt)|psi> (With field along z)
stepNum = int(p.args['ttotal']/p.args['tau'])
iterlist = np.linspace(0, p.args['hz'], num = stepNum+1, dtype=float)
iterlist = np.delete(iterlist, 0)
timestep = args['ttotal']/stepNum
print("Adiabatic time evolution")
for hz in tqdm(iterlist):
    args['hz'] = hz
    gateList = gates.makeGateList(args['real_n'], args)
    psi = mps.gateTEvol(psi, gateList, timestep, args['tau'], args=args)
tend = time.perf_counter()

# create result directory
# get system time
nowtime = datetime.datetime.now()
result_dir = '_'.join(['result_mps', str(nowtime.year), str(nowtime.month), 
str(nowtime.day), str(nowtime.hour), str(nowtime.minute)])
os.makedirs(result_dir, exist_ok=True)

# save parameters
with open(result_dir + '/parameters.txt', 'w+') as file:
    file.write(json.dumps(args))  # use json.loads to do the reverse
with open(result_dir + '/parameters.txt', 'a+') as file:
    file.write("\nAdiabatic evolution: " + str(tend-tstart) + " s\n")
    file.write('\nUsing String Operators:\n')

# apply undressed string
# <psi| exp(+iHt) S exp(-iHt) |psi>
for string_no in range(len(line)):
# for string_no in [-2]:
    bond_on_str, str_area = lat.convertToStrOp(ast.literal_eval(line[string_no]))
    # create string operator
    str_op = []
    for i in range(args['real_n']):
        str_op.append(np.reshape(p.iden, (1,2,2,1)))
    # convert coordinate to unique number in 1D
    bond_list = []
    for bond in bond_on_str:
        bond_list.append(lat.lat(bond[0:2], bond[2], (args['nx'],args['ny']), args['xperiodic']))
    for i in bond_list:
        str_op[i] = np.reshape(p.sx, (1,2,2,1))
    result = mps.matElem(psi, str_op, psi)
    with open(result_dir + '/parameters.txt', 'a+') as file:
        file.write('\n')
        file.write(str(str_area) + '\t' + str(bond_on_str) + '\t' + str(bond_list) + '\n')    # bonds
    with open(result_dir + '/undressed_result.txt', 'a+') as file:
        file.write(str(str_area) + "\t" + str(result) + '\n')

# quasi adiabatic evolution:
# exp(+iH't) exp(-iHt) |psi>
# stepNum = int(p.args['ttotal']/p.args['tau'])
# itlist = np.linspace(0, p.args['hz'], num = stepNum+1, dtype=float)
# iterlist = np.zeros(stepNum, dtype=float)
# for i in range(stepNum):
#     iterlist[i] = (itlist[i] + itlist[i+1])/2
# args['U'] *= -1
# args['g'] *= -1
# timestep = args['ttotal']/stepNum
# for hz in reversed(iterlist):
#     args['hz'] = -hz
#     gateList = gates.makeGateList(args['real_n'], args)
#     psi = mps.gateTEvol(psi, gateList, timestep, args['tau'], args=args)
# tend = time.perf_counter()
# with open(result_dir + '/parameters.txt', 'a+') as file:
#     file.write("\nQuasi-adiabatic evolution: " + str(tend-tstart) + " s\n")
#     file.write("Using " + str(stepNum) + ' steps (improved field in each step)\n')

# # apply dressed string
# # <psi| exp(+iHt) exp(-iH't) S exp(+iH't) exp(-iHt) |psi>
# for string_no in range(len(line)):
# # for string_no in [20,21]:
#     bond_on_str, str_area = lat.convertToStrOp(ast.literal_eval(line[string_no]))
#     # create string operator
#     str_op = []
#     for i in range(args['n']):
#         str_op.append(np.zeros((1,2,2,1), dtype=complex))
#     # convert coordinate to unique number in 1D
#     bond_list = []
#     for bond in bond_on_str:
#         bond_list.append(lat.lat(bond[0:2],bond[2],(args['nx'],args['ny']), args['xperiodic']))
#     for i in range(p.n):
#         if i in bond_list:
#             str_op[i][0,:,:,0] = p.sx
#         else:
#             str_op[i][0,:,:,0] = p.iden
#     result = mps.matElem(psi, str_op, psi)
#     with open(result_dir + '/dressed_result.txt', 'a+') as file:
#         file.write(str(str_area) + "\t" + str(result) + '\n')