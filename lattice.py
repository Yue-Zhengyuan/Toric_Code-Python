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
from itertools import product

def lat(site, dir, size, xperiodic):
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
    xperiodic : boolean
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

def vertex(site, args):
    """
    generate vertex centered at site (x, y)

    Parameters
    --------------
    site : int
        (x, y) coordinate of the center of the vertex
        (from 0 to size - 1)
    """
    nx, ny = args['nx'], args['ny']
    xperiodic = args['xperiodic']
    if xperiodic == True:
        sites = []
    elif xperiodic == False:
        sites = []
    return sites

def site(coord, size, xperiodic):
    """
    numbering the square lattice sites

    Parameters
    --------------
    site : (x, y)
        (x, y) coordinate of the site
    size : (int, int) -> (nx, ny)
        linear size of the square lattice
    xperiodic : boolean
        indicates PBC along x
    """
    nx, ny = size[0], size[1]
    x, y = coord[0], coord[1]
    if xperiodic == False:
        return y * nx + x
    else:
        if x == nx - 1:
            return y * (nx - 1)
        else:
            return y * (nx - 1) + x

def sitesOnBond(bond, size, xperiodic):
    """
    Return the number of sites connected by the bond

    Parameters
    -----------------
    bond: (x, y, 'r/d')
        the bond in question
    size : (int, int) -> (nx, ny)
        linear size of the square lattice
    xperiodic : boolean
        indicates PBC along x
    """
    x, y = bond[0], bond[1]
    if bond[2] == 'r':
        sites = [(x,y), (x+1,y)]
    elif bond[2] == 'd':
        sites = [(x,y), (x,y+1)]
    return site(sites[0],size,xperiodic), site(sites[1],size,xperiodic)

def makeBondList(size, xperiodic):
    """
    create bond list in the square lattice

    Parameters
    --------------
    size : (int, int)
        linear size of the square lattice
    xperiodic : boolean
        indicates PBC along x
    """
    # bondlist format
    # (0,1,2) -> (x,y,'r'/'d')
    # 3 -> bond number
    # 4,5 -> number of the two sites connected by the bond
    bondlist = []
    if xperiodic == False:
        for y, x in product(range(size[1]), range(size[0])):
            if y != size[1] - 1:
                site1, site2 = sitesOnBond((x,y,'d'), size, xperiodic)
                bondlist.append((x, y, 'd', lat((x,y),'d',size,xperiodic), site1, site2))
            if x != size[0] - 1:
                site1, site2 = sitesOnBond((x,y,'r'), size, xperiodic)
                bondlist.append((x, y, 'r', lat((x,y),'r',size,xperiodic), site1, site2))
    elif xperiodic == True:
        for y, x in product(range(size[1]), range(size[0])):
            if (y != size[1] - 1 and x != size[0] - 1):
                site1, site2 = sitesOnBond((x,y,'d'), size, xperiodic)
                bondlist.append((x, y, 'd', lat((x,y),'d',size,xperiodic), site1, site2))
            if x != size[0] - 1:
                site1, site2 = sitesOnBond((x,y,'r'), size, xperiodic)
                bondlist.append((x, y, 'r', lat((x,y),'r',size,xperiodic), site1, site2))
    bondlist = np.asarray(bondlist, 
    dtype=[('x', 'i4'), ('y', 'i4'), ('dir', 'U1'), ('bnum', 'i4'), ('s1', 'i4'), ('s2', 'i4')])
    return bondlist
    