# 
#   mpo_adiab_evol.py
#   Toric_Code-Python
#   adiabatic evolution of the string operator MPO
#
#   created on Apr 27, 2019 by Yue Zhengyuan
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
    args['hz'] = float(sys.argv[4])

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
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    result_dir = "mpopair_adiab_" + nowtime + "/"
    # use default p.args['nx']
    sep = 10
    # use default p.args['hz']
    os.makedirs(result_dir, exist_ok=True)
    # save parameters
    with open(result_dir + '/parameters.txt', 'w+') as file:
        pass

# create string operator pair (x-PBC) MPO enclosing different area
closed_str_list = crt.str_create3(args, sep)
string = closed_str_list[0]
bond_on_str, area, circum = crt.convertToStrOp(string, args)
bond_list = [lat.lat(bond_on_str[i][0:2], bond_on_str[i][2], (args['nx'], args['ny']), args['xperiodic']) for i in range(len(bond_on_str))]
# region = crt.selectRegion(bond_on_str, 1, args)

str_op = []
for i in range(args['real_n']):
    str_op.append(np.reshape(p.iden, (1,2,2,1)))
for i in bond_list:
    str_op[i] = np.reshape(p.sx, (1,2,2,1))

hz_max = args['hz']
# modify ttotal
args['ttotal'] = p.args['ttotal'] * hz_max/p.args['hz']
args['ttotal'] = np.around(args['ttotal'], decimals=2)

# save parameters
with open(result_dir + '/parameters.txt', 'a+') as file:
    file.write(json.dumps(args) + '\n')  # use json.loads to do the reverse
    # Output information
    file.write("area: {}\ncircumference: {}\n".format(area, circum))
    file.write("bond on str: \n{}\n".format(bond_on_str))
    file.write("bond number: \n{}\n".format(bond_list))

# adiabatic evolution (Heisenberg picture): 
# exp(+iH't) S exp(-iH't) - hz decreasing
# stepNum = int(args['ttotal']/args['tau'])
# iterlist = np.linspace(0, hz_max, num = stepNum+1, dtype=float)
# iterlist = np.delete(iterlist, 0)
# iterlist = np.flip(iterlist)
# timestep = args['ttotal']/stepNum
# for hz in tqdm(iterlist):
#     args['hz'] = hz
#     gateList = gates.makeGateList(len(str_op), args)
#     str_op = mpo.gateTEvol(str_op, gateList, timestep, timestep, args=args)

stepNum = int(args['ttotal']/args['tau'])
print("Step Number for Evolution:", stepNum)
iterlist = np.linspace(0, hz_max, num = stepNum+1, dtype=float)
iterlist = np.delete(iterlist, 0)
iterlist = np.flip(iterlist)
timestep = args['ttotal']/stepNum
for hz in tqdm(iterlist):
    args['hz'] = hz
    gateList = gates.makeGateList(len(str_op), args)
    str_op = mpo.gateTEvol(str_op, gateList, timestep, timestep, args=args)

# save string operator
op_save = np.asarray(str_op)
filename = "adiab_op_{}by{}_sep-{}_t-{:.2f}".format(args['nx'], args['ny'], sep, args['ttotal'])
np.save(result_dir + filename, op_save)
