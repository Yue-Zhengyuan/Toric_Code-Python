# 
#   run_mps_adiab.py
#   Toric_Code-Python
#   execute mps_adiab_evol.py
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
result_dir = "mps_adiab_" + nowtime + "/"
os.makedirs(result_dir, exist_ok=True)
out_dir = result_dir + 'outfile/'
os.makedirs(out_dir, exist_ok=True)

# create parameter file
parafile = result_dir + 'parameters.txt'
with open(parafile, 'w+') as file:
    pass

python = "~/anaconda3/bin/python"
nx_list = range(6, 7)
for nx in nx_list:
    # command parameters
    # 0 -> result dir
    # 1 -> system size nx
    # 2 -> outfile dir
    command = python + " mps_adiab_evol.py {0} {1} \
        > {2}outfile_{1} 2>&1 &".format(result_dir, nx, out_dir)
    os.system(command)

# if no error occurs, remove the outfiles
# condapy3 main_mps_pbc.py > outfile_mps_pbc 2>&1 &
# python main_mpo_adiab.py > outfile_mpoadiab 2>&1 &
