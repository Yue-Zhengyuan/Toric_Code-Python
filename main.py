# 
#   main.py
#   Toric_Code-Python
#   apply the gates and MPO to MPS
#
#   created on Jan 21, 2019 by Yue Zhengyuan
#

import numpy as np
import gates
import para_dict as p
import mps
import mpo
import sys
import copy
import time
import datetime
import os
import json

cutoff = 100
bondm = 32
para = p.para
#mode = sys.argv[1]
mode = '2'

# create directory to save result
# create folder to store results
# get system time
current_time = datetime.datetime.now()
result_dir = 'result'
result_dir += '_' + str(current_time.year)
result_dir += '-' + str(current_time.month)
result_dir += '-' + str(current_time.day)
result_dir += '_' + str(current_time.hour)
result_dir += '_' + str(current_time.minute)
os.makedirs(result_dir, exist_ok=True)

# save parameter of current running
with open(result_dir + '/parameters.txt', 'w+') as file:
    file.write(json.dumps(p.para)) # use `json.loads` to do the reverse

# create string operator MPO (S)
# labelling: [site][L vir leg, R vir leg, U phys leg, D phys leg]
# len(MPO): number of sites
str_op = []
for i in range(p.n):
    str_op.append(np.zeros((1,2,2,1), dtype=complex))
sites_on_str = [60,61,71,90,109,128,138,139]

for i in range(p.n):
    if i in sites_on_str:
        str_op[i][0,:,:,0] = p.sx
    else:
        str_op[i][0,:,:,0] = p.iden

# adiabatic continuation of string operator
if mode == '0':
    adiab_op = copy.copy(str_op)
    stepNum = int(p.para['ttotal'] / p.para['tau'])
    for hx in np.linspace(0, -p.para['hx'], num=stepNum, dtype=float):
        para['hx'] = hx
        gateList = gates.makeGateList(str_op, para)
        adiab_op = mpo.gateTEvol(adiab_op, gateList, para['tau'], para['tau'], cutoff, bondm)
    mpo.save_to_file(adiab_op, result_dir + '/adiab_op.txt')

# quasi-adiabatic continuation of string operator
elif mode == '1':
    quasi_op = copy.copy(str_op)
    stepNum = 5
    iter_list = np.linspace(0, -p.para['hx'], num=stepNum+1, dtype=float)
    iter_list = np.delete(iter_list, 0)
    for hx in iter_list:
        para['hx'] = hx
        gateList = gates.makeGateList(str_op, para)
        quasi_op = mpo.gateTEvol(quasi_op, gateList, para['ttotal']/stepNum, para['tau'], cutoff, bondm)
    mpo.save_to_file(quasi_op, result_dir + '/quasi_op.txt')

# no-field Heisenberg evolution of string operator
elif mode == '2':
    no_field_op = copy.copy(str_op)
    para['hx'] = 0.0
    gateList = gates.makeGateList(str_op, para)
    no_field_op = mpo.gateTEvol(no_field_op, gateList, para['ttotal'], para['tau'], cutoff, bondm)
    mpo.save_to_file(no_field_op, result_dir + '/no_field_op.txt')
