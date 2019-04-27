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
from str_create import str_create,str_create2,selectRegion, convertToStrOp
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
nx = int(sys.argv[2])
psi = np.load(result_dir + "/psi" + str(nx) + ".npy")
psi = list(psi)

args['nx'] = nx
n = 2 * (args['nx'] - 1) * args['ny']
# Y-non-periodic
n -= args['nx'] - 1
# X-non-periodic
n += args['ny'] - 1
args['n'] =  n
# n in case of periodic X
if args['xperiodic'] == True:
    args['real_n'] = n - (args['ny'] - 1)
else:
    sys.exit("Accepts x_PBC only.")
    args['real_n'] = n

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
    gateList = gates.makeGateList(args['real_n'], args)
    psi = mps.gateTEvol(psi, gateList, timestep, args['tau'], args=args)
tend = time.perf_counter()

# save new ground state
psi_save = np.asarray(psi)
np.save(result_dir + "/quasi_all_psi" + str(nx) + ".npy", psi_save)