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

args = copy.copy(p.args)
# clear magnetic field
args['hx'] = 0
args['hy'] = 0
args['hz'] = 0
# mode = sys.argv[1]
mode = '2'

# create string operator MPO (S)
# labelling: [site][L vir leg, R vir leg, U phys leg, D phys leg]
# len(MPO): number of sites
str_op = []
for i in range(p.n):
    str_op.append(np.zeros((1,2,2,1), dtype=complex))
# closed string
bond_on_str = [(2,2,'r'),(3,2,'d'),(2,3,'r'),(2,2,'d')]
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
psi = gnd_state.gnd_state_builder(args)

# exp(-iHt)|psi> (No field)
tstart = time.perf_counter()
gateList = gates.makeGateList(str_op, args)
psi = mps.gateTEvol(psi, gateList, args['ttotal'], args['tau'], args=args)
tend = time.perf_counter()
print("Time evolution:", tend-tstart, "s")

result = mps.matElem(psi, str_op, psi)

print("result =", result)