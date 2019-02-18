# 
#   lattice.py
#   Toric_Code-Python
#   convert 2D square lattice to 1D numbering
#
#   created on Feb 16, 2019 by Yue Zhengyuan
#

import numpy as np
import copy
import sys

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

def lat_table(size):
    """
    create numbering to square lattice bonds mapping table

    Parameters
    --------------
    size : int
        size of the square lattice
    """
    table = [[],[]]
    # bonds parallel to x
    # (i, j, 'r')
    for i in range(size - 1):
        for j in range(size):
            table[0].append((i,j,'r'))
            table[1].append(lat((i,j), 'r', size))

    # bonds parallel to y
    # (i, j, 'd')
    for i in range(size):
        for j in range(size - 1):
            table[0].append((i,j,'d'))
            table[1].append(lat((i,j), 'd', size))
    return table

def inv_lat(num, table):
    """
    convert bond number to coordinate according to 
    the conversion table created by lat_table

    Parameters
    --------------
    num : int
        number of the bond
    table : list
        number - coordinate conversion table
    """
    index = table[1].index(num)
    return table[0][index]