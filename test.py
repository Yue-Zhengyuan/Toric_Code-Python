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
import lattice

table = lattice.lat_table(p.para['nx'])
result = lattice.inv_lat(10, table)
print(result)