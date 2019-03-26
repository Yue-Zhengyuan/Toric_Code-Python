# 
#   main_mpo.py
#   Toric_Code-Python
#   apply the gates to MPO
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

args = copy.copy(p.args)
# clear magnetic field
args['hz'] = 0.0
# turn off vertex operator ZZZZ
args['U'] = 0
args['scale'] = True

# create result directory
# result_dir = sys.argv[1]

# create closed string operator MPO enclosing different area(S)
# read string from file
with open('newlist.txt', 'r') as f:
    line = f.readlines()
# string_no = int(sys.argv[2])
string_no = 19
bond_on_str, str_area = lat.convertToStrOp(ast.literal_eval(line[string_no]))
# create string operator
str_op = []
for i in range(args['n']):
    str_op.append(np.zeros((1,2,2,1), dtype=complex))
# convert coordinate to unique number in 1D
bond_list = []
for bond in bond_on_str:
    bond_list.append(lat.lat(bond[0:2],bond[2],args['nx']))
for i in range(p.n):
    if i in bond_list:
        str_op[i][0,:,:,0] = p.sx
    else:
        str_op[i][0,:,:,0] = p.iden

# create Toric Code ground state |psi>
# psi = gnd_state.gnd_state_builder(args)

tstart = time.perf_counter()

# quasi-adiabatic evolution: 
# exp(-iH't) S exp(+iH't)
stepNum = 25
iterlist = np.linspace(0, p.args['hz'], num = stepNum+1, dtype=float)
iterlist = np.delete(iterlist, 0)
timestep = args['ttotal']/stepNum
for hz in iterlist:
    args['hz'] = hz
    gateList = gates.makeGateList(str_op, args)
    str_op = mpo.gateTEvol(str_op, gateList, timestep, timestep, args=args)

tend = time.perf_counter()
print("Time evolution:", tend - tstart)
mpo.printdata(str_op)