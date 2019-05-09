# 
#   run_mpo_adiab.py
#   Toric_Code-Python
#   execute mpo_adiab_evol.py
#
#   created on Apr 24, 2019 by Yue Zhengyuan
#

import os, sys, time, datetime, json
import para_dict as p
import numpy as np
from copy import copy
from itertools import product

# create result directory
nowtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
result_dir = "mpopair_adiab-tevol_" + nowtime + "/"
os.makedirs(result_dir, exist_ok=True)
out_dir = result_dir + 'outfile/'
os.makedirs(out_dir, exist_ok=True)

# create parameter file
parafile = result_dir + 'parameters.txt'
with open(parafile, 'w+') as file:
    pass

python = "~/anaconda3/bin/python"
# create string list (can handle both x-PBC and OBC)
sep_list = [6, 10, 13]
nx_list = range(3, 8)
hz_list = np.linspace(0, p.args['hz'], num=11, endpoint=True)
hz_list = np.delete(hz_list, 0)
for nx, sep in product(nx_list, sep_list):
    if (nx == 6 and sep == 10):
        hz_list = np.linspace(0, p.args['hz'], num=11, endpoint=True)
        hz_list = np.delete(hz_list, 0)
    else:
        hz_list = [p.args['hz']]
    for hz in hz_list:
    # command parameters
    # 0 -> result dir
    # 1 -> system size nx
    # 2 -> string separation
    # 3 -> max hz
    # 4 -> outfile dir
        command = python + " mpo_adiab_evol.py {0} {1} {2} {3:.2f} > {4}outfile_{1}_{2}_{3:.2f} 2>&1 &".format(result_dir, nx, sep, hz, out_dir)
        os.system(command)
# command format
# python main_mpo_adiab.py > outfile 2>&1 &
