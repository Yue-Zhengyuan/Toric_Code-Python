# 
#   mpo_bonddim.py
#   Toric_Code-Python
#   extract virtual bond dimension in MPO
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
nx = 6
ny = args['ny']
sep = 10
hz = args['hz']

result_dir = "mpopair_adiab-tevol_2019-05-08_21-44/"
label = "adiab"
t_list = np.linspace(0, p.args['ttotal'], num=11, endpoint=True)
t_list = np.delete(t_list, 0)
for t in t_list:
# for t in [0.6]:
    op = np.load(result_dir + '{}_op_{}by{}_sep-{}_t-{:.2f}.npy'.format(label, nx, ny, sep, t))
    op = list(op)
    bond_file = result_dir + '/bond_{}_{}by{}_sep-{}_t-{:.2f}.txt'.format(label, nx, ny, sep, t)
    with open(bond_file, 'w+') as file:
        pass

    for site in tqdm(range(len(op))):
        bonddim = np.shape(op[site])[-1]
        with open(bond_file, 'a+') as file:
            file.write("{}\t{}\n".format(site, bonddim))
