# 
#   lattice.py
#   Toric_Code-Python
#   convert 2D square lattice to 1D numbering
#
#   created on Mar 23, 2019 by Yue Zhengyuan
#

import numpy as np
import copy
import sys

def lat(site, dir, size, xperiodic=True):
    """
    numbering the square lattice bonds

    Parameters
    --------------
    site : int
        (x, y) coordinate of one endpoint of the bond
        (from 0 to size - 1)
    dir : 'r'/'d'
        direction of the bond (right or downward)
    size : (int, int) -> (nx, ny)
        linear size of the square lattice
    xperiodic : default True
        indicates PBC along x
    """
    nx, ny = size[0], size[1]
    x, y = site[0], site[1]
    if xperiodic == False:
        step = 2 * nx - 1
        if dir == 'r':
            line_start = y * step + 1
            num = line_start + x
            num -= 1
        elif dir == 'd':
            col_start = nx + x
            num = col_start + y * step
            num -= 1
        else:
            sys.exit('Wrong direction of the bond')
    if xperiodic == True:
        if dir == 'r':
            num = y * 2 * (nx - 1) + x
        elif dir == 'd':
            if x == nx - 1:
                num = (y * 2 + 1) * (nx - 1)
            else:
                num = (y * 2 + 1) * (nx - 1) + x
        else:
            sys.exit('Wrong direction of the bond')
    return num

def selectRegion(string, width, size, xperiodic=True):
    """
    select a region within width from the given string

    Parameters
    -----------------
    string : list of bonds
        bonds on the string operator
    width : int
        width of the spreading from string operator
    size : (int, int)
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
            elif x == size[0]  - 1:    # on the right boundary; add left plaquette only
                right = []                
            for newbond in left: 
                region.append(lat(newbond[0:2],newbond[2],size, xperiodic=xperiodic))
            for newbond in right:
                region.append(lat(newbond[0:2],newbond[2],size, xperiodic=xperiodic))
        if dir == 'r':
            # add the plaquette above/below this bond
            above = [(x,y-1,'r'),(x,y-1,'d'),(x+1,y-1,'d'),(x,y,'r')]
            below = [(x,y,'r'),(x,y,'d'),(x+1,y,'d'),(x,y+1,'r')]
            if y == 0:          # on the upper boundary; add plaquette below only
                above = []
            if y == size[1] - 1:    # on the lower boundary; add plaquette above only
                below = []                
            for newbond in above: 
                region.append(lat(newbond[0:2],newbond[2],size, xperiodic=xperiodic))
            for newbond in below:
                region.append(lat(newbond[0:2],newbond[2],size, xperiodic=xperiodic))
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
        if ((x, y-1) not in plist):      # upper plaquette
            bond_on_str.append((x,y,'r'))
        if ((x, y+1) not in plist):      # lower plaquette
            bond_on_str.append((x,y+1,'r'))
        if ((x-1, y) not in plist):      # left plaquette
            bond_on_str.append((x,y,'d'))
        if ((x+1, y) not in plist):      # right plaquette
            bond_on_str.append((x+1,y,'d'))
    return bond_on_str, area