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
    size : (int, int) -> (nx, ny)
        linear size of the square lattice
    """
    nx, ny = size[0], size[1]
    x, y = site[0], site[1]
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
    return num

def site(coord, size):
    """
    numbering the square lattice sites

    Parameters
    --------------
    coord : (x, y)
        (x, y) coordinate of the site
    size : (int, int) -> (nx, ny)
        linear size of the square lattice
    """
    return coord[1] * size[0] + coord[0]

def inv_site(num, size):
    """
    convert site number to coordinate

    Parameters
    --------------
    num : int
        number of the site
    size : (int, int) -> (nx, ny)
        linear size of the square lattice
    """
    y = int(num / size[0])
    x = num % size[0]
    return (x, y)

def sitesOnBond(bond, size):
    """
    Return the number of sites connected by the bond

    Parameters
    -----------------
    bond: (x, y, 'r/d')
        the bond in question
    size : (int, int) -> (nx, ny)
        linear size of the square lattice
    """
    x, y = bond[0], bond[1]
    if bond[2] == 'r':
        sites = [(x,y), (x+1,y)]
    elif bond[2] == 'd':
        sites = [(x,y), (x,y+1)]
    return site(sites[0],size), site(sites[1],size)

def makeBondList(size):
    """
    create bond list in the square lattice

    Parameters
    --------------
    size : (int, int)
        linear size of the square lattice
    """
    # bondlist format
    # (0,1,2) -> (x,y,'r'/'d')
    # 3 -> bond number
    # 4,5 -> number of the two sites connected by the bond
    bondlist = []
    for y in range(size[1]):
        for x in range(size[0]):
            if y != size[1]-1:
                site1, site2 = sitesOnBond((x,y,'d'), size)
                bondlist.append((x, y, 'd', lat((x,y),'d',size), site1, site2))
            if x != size[0]-1:
                site1, site2 = sitesOnBond((x,y,'r'), size)
                bondlist.append((x, y, 'r', lat((x,y),'r',size), site1, site2))
    bondlist = np.asarray(bondlist, 
    dtype=[('x', 'i4'), ('y', 'i4'), ('dir', 'U1'), ('bnum', 'i4'), ('s1', 'i4'), ('s2', 'i4')])
    return bondlist

def selectRegion(string, width, size):
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
                region.append(lat(newbond[0:2], newbond[2], size))
            for newbond in right:
                region.append(lat(newbond[0:2], newbond[2], size))
        if dir == 'r':
            # add the plaquette above/below this bond
            above = [(x,y-1,'r'),(x,y-1,'d'),(x+1,y-1,'d'),(x,y,'r')]
            below = [(x,y,'r'),(x,y,'d'),(x+1,y,'d'),(x,y+1,'r')]
            if y == 0:          # on the upper boundary; add plaquette below only
                above = []
            if y == size[1] - 1:    # on the lower boundary; add plaquette above only
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