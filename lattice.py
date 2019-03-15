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
        linear size of the square lattice
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
        linear size of the square lattice
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
        linear size of the lattice system
    """
    region = []
    for bond in string:
        x, y, dir = bond[0], bond[1], bond[2]
        if dir == 'd':
            # add the plaquette to the left/right of this bond
            left = [(x-1,y,'r'),(x-1,y,'d'),(x,y,'d'),(x-1,y+1,'r')]
            right = [(x,y,'d'),(x,y,'r'),(x+1,y,'d'),(x,y+1,'r')]
            # check whether this is a boundary bond
            if x == 0:              # on the left boundary; add right plaquette only
                left = []
            elif x == size  - 1:    # on the right boundary; add left plaquette only
                right = []                
            for newbond in left: 
                region.append(lat(newbond[0:2], newbond[2], size))
            for newbond in right:
                region.append(lat(newbond[0:2], newbond[2], size))
        if dir == 'r':
            # add the plaquette above/below this bond
            above = [(x,y-1,'r'),(x,y-1,'d'),(x+1,y-1,'d'),(x,y,'r')]
            below = [(x,y,'r'),(x,y,'d'),(x+1,y,'d'),(x,y+1,'r')]
            if y == 0:          # on the upper boundary; add plaquette below only
                above = []
            if y == size - 1:    # on the lower boundary; add plaquette above only
                below = []                
            for newbond in above: 
                region.append(lat(newbond[0:2], newbond[2], size))
            for newbond in below:
                region.append(lat(newbond[0:2], newbond[2], size))
    region = np.asarray(region, dtype=int)
    region = np.unique(region)
    return region

def convertToStrOp(plist):
    """
    Convert the plaquettes enclosed by string to the string
    Each plaquette is labeled by the coordinate of its upper-left corner
    """
    area = len(plist)
    plist = [tuple(plq) for plq in plist]
    bond_on_str = []
    for plq in plist:
        x, y = plq[0], plq[1]
        if not((x, y-1) in plist):      # upper plaquette
            bond_on_str.append((x,y,'r'))
        if not((x, y+1) in plist):      # lower plaquette
            bond_on_str.append((x,y+1,'r'))
        if not((x-1, y) in plist):      # left plaquette
            bond_on_str.append((x,y,'d'))
        if not((x+1, y) in plist):      # right plaquette
            bond_on_str.append((x+1,y,'d'))
    return bond_on_str, area