# 
#   mpo_quasi_evol.py
#   Toric_Code-Python
#   quasi-adiabatic evolution of the string operator MPO
#
#   created on Apr 29, 2019 by Yue Zhengyuan
#

import numpy as np
import gates, mps, mpo, gnd_state
import para_dict as p
import lattice as lat
import str_create as crt
import os, sys, time, datetime, json, ast
from copy import copy
from tqdm import tqdm

args = copy(p.args)

# get arguments from command line
if len(sys.argv) > 1:   # executed by the "run..." file
    result_dir = sys.argv[1]
    args['nx'] = int(sys.argv[2])
    sep = int(sys.argv[3])
    benchmark = bool(int(sys.argv[4]))
    width = 3
    # modify total number of sites
    n = 2 * (args['nx'] - 1) * args['ny']
    # Y-non-periodic
    n -= args['nx'] - 1
    # X-non-periodic
    n += args['ny'] - 1
    args['n'] = n
    # n in case of periodic X
    if args['xperiodic'] == True:
        args['real_n'] = n - (args['ny'] - 1)
    else:
        args['real_n'] = n
else:
    benchmark = True
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    if benchmark == False:
        result_dir = "mpopair_quasi-tevol_" + nowtime + "/"
    elif benchmark == True:
        result_dir = "mpopair_bm-tevol_" + nowtime + "/"
    # use default p.args['nx']
    sep = 10
    # use default p.args['hz']
    width = 3
    os.makedirs(result_dir, exist_ok=True)
    # save parameters
    with open(result_dir + '/parameters.txt', 'w+') as file:
        pass

# create string operator pair (x-PBC) MPO enclosing different area
closed_str_list = crt.str_create3(args, sep)
string = closed_str_list[0]
bond_on_str, area, circum = crt.convertToStrOp(string, args)
bond_list = [lat.lat(bond_on_str[i][0:2], bond_on_str[i][2], (args['nx'], args['ny']), args['xperiodic']) for i in range(len(bond_on_str))]
region = crt.selectRegion2(bond_on_str, width, args)

str_op = []
for i in range(args['real_n']):
    str_op.append(np.reshape(p.iden, (1,2,2,1)))
for i in bond_list:
    str_op[i] = np.reshape(p.sx, (1,2,2,1))

# save parameters
with open(result_dir + '/parameters.txt', 'a+') as file:
    file.write(json.dumps(args) + '\n')  # use json.loads to do the reverse
    # Output information
    file.write("area: {}\ncircumference: {}\n".format(area, circum))
    file.write("bond on str: \n{}\n".format(bond_on_str))
    file.write("bond number: \n{}\n".format(bond_list))
    if benchmark == False:
        file.write("bond within distance {}: \n{}\n\n".format(width, region))

# quasi-adiabatic evolution (using Hamiltonian in the selected region): 
# S' = exp(-iH't) S exp(+iH't) - hz increasing
hz_max = args['hz']
stepNum = int(args['ttotal']/args['tau'])
iterlist = np.linspace(0, hz_max, num = stepNum+1, dtype=float)
iterlist = np.delete(iterlist, 0)
timestep = args['ttotal']/stepNum
args['g'] *= -1
step = 0
for hz in tqdm(iterlist):
    args['hz'] = -hz
    if benchmark == False:
        gateList = gates.makeGateList(len(str_op), args, region=region)
    elif benchmark == True:
        gateList = gates.makeGateList(len(str_op), args)
    str_op = mpo.gateTEvol(str_op, gateList, timestep, timestep, args=args)
    step += 1
    # save string operator
    if step % 20 == 0:
        t = step / stepNum * args['ttotal']
        op_save = np.asarray(str_op)
        filename = "quasi_op_{}by{}_sep-{}_hz-{:.2f}_t-{:.2f}".format(args['nx'], args['ny'], sep, hz_max, t)
        np.save(result_dir + filename, op_save)

# adiabatic evolution (Heisenberg picture): 
# exp(+iH't) S exp(-iH't) - hz decreasing
stepNum = int(args['ttotal']/args['tau'])
iterlist = np.linspace(0, hz_max, num = stepNum+1, dtype=float)
iterlist = np.delete(iterlist, 0)
iterlist = np.flip(iterlist)
timestep = args['ttotal']/stepNum
# restore g
args['g'] *= -1
for hz in tqdm(iterlist):
    args['hz'] = hz
    gateList = gates.makeGateList(len(str_op), args)
    str_op = mpo.gateTEvol(str_op, gateList, timestep, timestep, args=args)
# save string operator
op_save = np.asarray(str_op)
filename = "final_op_{}by{}_sep-{}_hz-{:.2f}".format(args['nx'], args['ny'], sep, hz_max)
np.save(result_dir + filename, op_save)
