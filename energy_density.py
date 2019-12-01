# 
#   energy_density.py
#   Toric_Code-Python
#   calculate energy distribution for state
#   to detect quasi-particle excitation
#
#   created on Jun 3, 2019 by Yue Zhengyuan
#

import numpy as np
import gates, mps, mpo, gnd_state
import para_dict as p
import lattice as lat
import str_create as crt
import os, sys, time, datetime, json, ast
from copy import copy
from tqdm import tqdm

# electric charge is located at the stars
def get_energy_dens(site, args):
    """
    calculate local energy density

    Parameters
    --------------
    site : int
        (x, y) coordinate of one endpoint of the bond
        (from 0 to size - 1)
    """
    U, g = args['U'], args['g']
    nx, ny = args['nx'], args['ny']
    return 0