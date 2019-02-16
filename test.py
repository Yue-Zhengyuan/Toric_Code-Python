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

def lat(site, dir, size):
    """
    numbering the square lattice bonds

    Parameters
    --------------
    site : int
        (x, y) coordinate of one endpoint of the bond
        (from 0 to size - 1)
    dir : 'r'/'d'
        direction of the bond (right or downward)
    size : int
        size of the square lattice
    """
    step = 2*size - 1
    if dir == 'r':
        line_start = site[1] * step + 1
        num = line_start + site[0]
        num -= 1

    elif dir == 'd':
        col_start = size + site[0]
        num = col_start + site[1] * step
        num -= 1

    else:
        sys.exit('Wrong direction of the bond')
    # will add boundary check later
    return num

print(lat((2,2), 'd', 5))