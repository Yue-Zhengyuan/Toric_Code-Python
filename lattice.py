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
    for j in range(size):
        for i in range(size):
            if j != size-1:
                table[0].append((i,j,'d'))
                table[1].append(lat((i,j), 'd', size))
            if i != size-1:
                table[0].append((i,j,'r'))
                table[1].append(lat((i,j), 'r', size))
    
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

def selectRegion(string, width, size):
    """
    select a region within width from the given string

    Parameters
    -----------------
    string : list of bonds
        bonds on the string operator
    width : int
        width of the spreading from string operator
    size : int
        size of the lattice system
    """
    region = []
    for bond in string:
        if bond[2] == 'd':
            lowlimit = max(0, bond[0]-width)
            uplimit = min(size-1, bond[0]+width)
            for i in np.arange(lowlimit, uplimit, 1, dtype=int):
                region.append(lat((i, bond[1]), 'd', size))
        if bond[2] == 'r':
            lowlimit = max(0, bond[1]-width)
            uplimit = min(size-1, bond[1]+width)
            for i in np.arange(lowlimit, uplimit, 1, dtype=int):
                region.append(lat((i, bond[1]), 'r', size))
    region = np.asarray(region, dtype=int)
    region = np.unique(region)
    return region