# 
#   mpo_entg.py
#   Toric_Code-Python
#   apply the Trotter gates to MPS
#
#   created on Feb 18, 2019 by Yue Zhengyuan
#

import numpy as np
import mps, mpo, gnd_state
import para_dict as p
import lattice as lat
import sys, os, datetime, time, json, ast
from copy import copy
from tqdm import tqdm
from itertools import product

args = copy(p.args)
nx = 4
ny = args['ny']
sep = 10
hz = args['hz']

result_dir = "mpopair_quasi-tevol_2019-05-09_19-30/"
label = "quasi"
t_list = np.linspace(0, p.args['ttotal'], num=11, endpoint=True)
t_list = np.delete(t_list, 0)
for t in [1.0]:
    op = np.load(result_dir + '{}_op_{}by{}_sep-{}_hz-{:.2f}_t-{:.2f}.npy'.format(label, nx, ny, sep, hz, t))
    op = list(op)
    entg_file = result_dir + '/entg_{}_{}by{}_sep-{}_t-{:.2f}.txt'.format(label, nx, ny, sep, t)
    with open(entg_file, 'w+') as file:
        pass

    for site in tqdm(range(len(op))):
        op, entg = mpo.position(op, site, args, oldcenter=site - 1, compute_entg=True)
        y = int(site / (nx - 1))
        x = site % (nx - 1)
        if y % 2 == 0:
            bonddir = 'r'
        else:
            bonddir = 'd'
        with open(entg_file, 'a+') as file:
            file.write("{}\t{}\t{}\t{}\t{}\n".format(site, x, int(y/2), bonddir, entg))
