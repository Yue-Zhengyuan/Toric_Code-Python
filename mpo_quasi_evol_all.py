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
closed_str_list = crt.str_create(args, args['ny'] - 1)
string = closed_str_list[0]
bond_on_str, area, circum = crt.convertToStrOp(string, args)
bond_list = [lat.lat(bond_on_str[i][0:2], bond_on_str[i][2], (args['nx'], args['ny']), args['xperiodic']) for i in range(len(bond_on_str))]
# create string operator
str_op = []
for i in range(args['real_n']):
    str_op.append(np.reshape(p.iden, (1,2,2,1)))
for i in bond_list:
    str_op[i] = np.reshape(p.sx, (1,2,2,1))
with open(result_dir + '/parameters.txt', 'a+') as file:
    # Output information
    file.write(str(area) + '\t' + str(circum) + '\n')
    file.write("bond on str: \n" + str(bond_on_str) + '\n')
    file.write("bond number: \n" + str(bond_list) + '\n')
    # file.write("bond within distance 1: \n" + str(region) + '\n')

tstart = time.perf_counter()

# quasi-adiabatic evolution: 
# exp(-iH't) S exp(+iH't)
stepNum = int(p.args['ttotal']/p.args['tau'])
iterlist = np.linspace(0, p.args['hz'], num = stepNum+1, dtype=float)
iterlist = np.delete(iterlist, 0)
timestep = args['ttotal']/stepNum
for hz in tqdm(iterlist):
    args['hz'] = hz
    gateList = gates.makeGateList(str_op, args)
    str_op = mpo.gateTEvol(str_op, gateList, timestep, timestep, args=args)

tend = time.perf_counter()
with open(result_dir + '/parameters.txt', 'a+') as file:
    file.write("\nAdiabatic evolution: " + str(tend-tstart) + " s\n")

# save string operator
op_save = np.asarray(str_op)
np.save(result_dir + "/quasi_op.npy", op_save)
