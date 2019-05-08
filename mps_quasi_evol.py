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
    width = 2
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
    result_dir = "mps_quasi_" + nowtime + "/"
    # use default p.args['nx']
    sep = 10
    # use default p.args['hz']
    width = 2
    os.makedirs(result_dir, exist_ok=True)
    benckmark = False
    # save parameters
    with open(result_dir + '/parameters.txt', 'w+') as file:
        pass

# create string operator pair (x-PBC) MPO enclosing different area
closed_str_list = crt.str_create3(args, sep)
string = closed_str_list[0]
bond_on_str, area, circum = crt.convertToStrOp(string, args)
bond_list = [lat.lat(bond_on_str[i][0:2], bond_on_str[i][2], (args['nx'], args['ny']), args['xperiodic']) for i in range(len(bond_on_str))]
region = crt.selectRegion2(bond_on_str, width, args)

# save parameters
with open(result_dir + '/parameters.txt', 'a+') as file:
    file.write(json.dumps(args) + '\n')  # use json.loads to do the reverse
    # Output information
    file.write("area: {}\ncircumference: {}\n".format(area, circum))
    file.write("bond on str: \n{}\n".format(bond_on_str))
    file.write("bond number: \n{}\n".format(bond_list))
    if benchmark == False:
        file.write("bond within distance {}: \n{}\n\n".format(width, region))

# create Toric Code ground state |psi>
psi = gnd_state.gnd_state_builder(args)

# adiabatic evolution: exp(-iHt)|psi> (With field along z)
hz_max = args['hz']
stepNum = int(args['ttotal']/args['tau'])
iterlist = np.linspace(0, hz_max, num = stepNum+1, dtype=float)
iterlist = np.delete(iterlist, 0)
timestep = args['ttotal']/stepNum
step = 0
for hz in tqdm(iterlist):
    args['hz'] = hz
    gateList = gates.makeGateList(args['real_n'], args)
    psi = mps.gateTEvol(psi, gateList, timestep, args['tau'], args=args)
    step += 1
    # save new ground state
    if step % int(stepNum / 4) == 0:
        psi_save = np.asarray(psi)
        filename = "mps_{}by{}_hz-{:.2f}".format(args['nx'], args['ny'], hz)
        np.save(result_dir + filename, psi_save)
    
# quasi-adiabatic evolution: exp(+iHt) exp(-iHt)|psi> (With field along z)
hz_max = args['hz']
stepNum = int(args['ttotal']/args['tau'])
iterlist = np.linspace(0, hz_max, num = stepNum+1, dtype=float)
iterlist = np.delete(iterlist, 0)
iterlist = np.flip(iterlist)
timestep = args['ttotal']/stepNum
step = 0
args['g'] *= -1
for hz in tqdm(iterlist):
    args['hz'] = -hz
    gateList = gates.makeGateList(args['real_n'], args, region=region)
    psi = mps.gateTEvol(psi, gateList, timestep, args['tau'], args=args)
    step += 1
psi_save = np.asarray(psi)
filename = "mps_final_{}by{}_hz-{:.2f}".format(args['nx'], args['ny'], hz)
np.save(result_dir + filename, psi_save)