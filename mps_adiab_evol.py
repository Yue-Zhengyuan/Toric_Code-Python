# 
#   mps_adiab_evol.py
#   Toric_Code-Python
#   create ground state (under perturbation) via adiabatic evolution
#
#   created on Feb 18, 2019 by Yue Zhengyuan
#

import numpy as np
import gates, mps, gnd_state
import para_dict as p
import lattice as lat
import os, sys, time, datetime, json, ast
from copy import copy
from tqdm import tqdm

args = copy(p.args)

# get arguments from command line
if len(sys.argv) > 1:   # executed by the "run..." file
    result_dir = sys.argv[1]
    args['nx'] = int(sys.argv[2])
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
    # save parameters
    with open(result_dir + '/parameters.txt', 'a+') as file:
        file.write(json.dumps(args) + '\n')  # use json.loads to do the reverse
else:
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    result_dir = "mps_adiab_" + nowtime + "/"
    # use default args['nx']
    os.makedirs(result_dir, exist_ok=True)
    # save parameters
    with open(result_dir + '/parameters.txt', 'w+') as file:
        file.write(json.dumps(args) + '\n\n')  # use json.loads to do the reverse

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
    