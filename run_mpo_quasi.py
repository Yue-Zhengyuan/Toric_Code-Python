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
benchmark = True
nowtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
if benchmark == False:
    result_dir = "mpopair_quasi-tevol_" + nowtime + "/"
elif benchmark == True:
    result_dir = "mpopair_bm-tevol_" + nowtime + "/"
os.makedirs(result_dir, exist_ok=True)
out_dir = result_dir + 'outfile/'
os.makedirs(out_dir, exist_ok=True)

# create parameter file
parafile = result_dir + 'parameters.txt'
with open(parafile, 'w+') as file:
    pass

python = "~/anaconda3/bin/python"
# create string list (can handle both x-PBC and OBC)
sep_list = [10]
nx_list = range(6, 7)
for nx, sep in product(nx_list, sep_list):
    # command parameters
    # 0 -> result dir
    # 1 -> system size nx
    # 2 -> string separation
    # 3 -> benchmark
    # 4 -> outfile dir
    command = python + " mpo_quasi_evol.py {0} {1} {2} {3} > {4}outfile_{1}_{2} 2>&1 &".format(result_dir, nx, sep, int(benchmark), out_dir)
    os.system(command)
# command format
# python main_mpo_adiab.py > outfile_mpoadiab 2>&1 &
