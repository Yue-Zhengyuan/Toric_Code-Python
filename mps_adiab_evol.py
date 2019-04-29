# 
#   state_adiab_evol.py
#   Toric_Code-Python
#   create ground state after turning on perturbation
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
from copy import copy
import time
import datetime
import os
import json
import ast
from tqdm import tqdm

args = copy(p.args)

result_dir = sys.argv[1]
# hz_max = p.args['hz']
hz_max = float(sys.argv[2])

# # create result directory
# # get system time
# nowtime = datetime.datetime.now()
# result_dir = '_'.join(['result_mps', str(nowtime.year), str(nowtime.month), 
# str(nowtime.day), str(nowtime.hour), str(nowtime.minute)])
# os.makedirs(result_dir, exist_ok=True)

# create Toric Code ground state |psi>
psi = gnd_state.gnd_state_builder(args)

# adiabatic evolution: exp(-iHt)|psi> (With field along z)
print("Adiabatic Evolution")
tstart = time.perf_counter()
stepNum = int(p.args['ttotal']/p.args['tau'])
iterlist = np.linspace(0, hz_max, num = stepNum+1, dtype=float)
iterlist = np.delete(iterlist, 0)
timestep = args['ttotal']/stepNum
for hz in tqdm(iterlist):
    args['hz'] = hz
    gateList = gates.makeGateList(args['real_n'], args)
    psi = mps.gateTEvol(psi, gateList, timestep, args['tau'], args=args)
tend = time.perf_counter()

# save parameters
with open(result_dir + '/parameters.txt', 'a+') as file:
    file.write(json.dumps(args) + '\n')  # use json.loads to do the reverse
# save new ground state
psi_save = np.asarray(psi)
np.save(result_dir + '/psi' + '_hz_' + str(hz_max), psi_save)
    