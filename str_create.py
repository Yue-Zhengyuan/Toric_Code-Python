# 
#   str_create.py
#   Toric_Code-Python
#   functions useful for string creation
#
#   created on Apr 10, 2019 by Yue Zhengyuan
#

import numpy as np
import lattice as lat
import sys
from copy import copy
import ast

def str_create(args, maxdy):
    """
    create closed string operator automatically from input nx and ny
    for systems long in y direction

    Parameters
    ----------------
    args : dictionary
        should include 'nx', 'ny', 'xperiodic' entries
    maxsep : int
        maximum separation of the closed string opeator (x-OBC)
        or x-string operator pair (x-PBC) along y direction

    Returns
    ----------------
    plaquettes enclosed by the closed string
    """
    if maxdy > args['ny'] - 1:
        sys.exit("max y separation larger than system y size")
    elif maxdy < 1:
        sys.exit("max y separation smaller than 1")

    closed_str_list = []
    plqlist = []
    # create string pair list by adding plaquette
    mid_y = int(args['ny'] / 2)
    mid_x = int(args['nx'] / 2)
    # initial value
    y1 = mid_y - 1
    y2 = mid_y - 1
    for str_sep in range(1, maxdy + 1):
        if str_sep == 1:
            j = y1
        elif str_sep % 2 == 0:
            y2 += 1
            j = y2
        elif str_sep % 2 == 1:
            y1 -= 1
            j = y1
            
        if args['xperiodic'] == True:
            iterlist = range(args['nx'] - 2)
        else:
            iterlist = range(args['nx'] - 1)
        for i in iterlist:
            plqlist.append([i, j])
        closed_str_list.append(copy(plqlist))
    return closed_str_list

def str_create2(args, maxdy):
    """
    create closed string operator automatically from input nx and ny
    for systems long in y direction

    Parameters
    ----------------
    args : dictionary
        should include 'nx', 'ny', 'xperiodic' entries
    maxsep : int
        maximum separation of the closed string opeator (x-OBC)
        or x-string operator pair (x-PBC) along y direction

    Returns
    ----------------
    plaquettes enclosed by the closed string
    """
    if maxdy > args['ny'] - 1:
        sys.exit("max y separation larger than system y size")
    elif maxdy < 1:
        sys.exit("max y separation smaller than 1")

    closed_str_list = []
    plqlist = []
    # create string pair list by adding plaquette
    mid_y = int(args['ny'] / 2)
    mid_x = int(args['nx'] / 2)
    # initial value
    y1 = mid_y - 1
    y2 = mid_y - 1
    for str_sep in range(1, maxdy + 1):
        if str_sep == 1:
            j = y1
        elif str_sep % 2 == 0:
            y2 += 1
            j = y2
        elif str_sep % 2 == 1:
            y1 -= 1
            j = y1

        if args['xperiodic'] == True:
            iterlist = range(args['nx'] - 2)
        else:
            iterlist = range(args['nx'] - 1)
        for i in iterlist:
            plqlist.append([i, j])
            closed_str_list.append(copy(plqlist))

    return closed_str_list

def str_create3(args, dy):
    """
    create closed string operator separated by dy from input nx and ny
    for systems long in y direction (PBC only)

    Parameters
    ----------------
    args : dictionary
        should include 'nx', 'ny', 'xperiodic' entries
    maxsep : int
        maximum separation of the closed string opeator (x-OBC)
        or x-string operator pair (x-PBC) along y direction

    Returns
    ----------------
    plaquettes enclosed by the closed string
    """
    if dy > args['ny'] - 1:
        sys.exit("max y separation larger than system y size")
    elif dy < 1:
        sys.exit("max y separation smaller than 1")

    if args['xperiodic'] == False:
        sys.exit("Accepts x-PBC only")

    closed_str_list = []
    plqlist = []
    # create string pair list by adding plaquette
    mid_y = int(args['ny'] / 2)
    mid_x = int(args['nx'] / 2)
    # initial value
    y1 = mid_y - 1
    y2 = mid_y
    str_sep = 1
    while str_sep <= dy:
        if str_sep == 1:
            pass
        elif str_sep % 2 == 0:
            y2 += 1
        elif str_sep % 2 == 1:
            y1 -= 1
        str_sep += 1
    for j in range(y1, y2):
        for i in range(args['nx'] - 1):
            plqlist.append([i, j])
    closed_str_list.append(copy(plqlist))
    return closed_str_list

def convertToStrOp(plqlist, args):
    """
    Convert the plaquettes enclosed by string to the string
    Each plaquette is labeled by the coordinate of its **UPPER-LEFT** corner

    Returns
    ----------------
    bond_on_str : bonds on the string (x, y, dir) \n
    area : enclosed area of the string \n
    circum : circumference of the string
    """
    xperiodic = args['xperiodic']
    nx, ny = args['nx'], args['ny']
    area = len(plqlist)
    plqlist = [tuple(plq) for plq in plqlist]
    bond_on_str = []

    for plq in plqlist:
        x, y = plq[0], plq[1]
        if (x < 0 or x >= nx - 1):
            sys.exit("Plaquette x out of range")

        if not((x, y-1) in plqlist):      # upper plaquette
            bond_on_str.append((x,y,'r'))
        if not((x, y+1) in plqlist):      # lower plaquette
            bond_on_str.append((x,y+1,'r'))

        if xperiodic == False:
            if not((x-1, y) in plqlist):      # left plaquette
                bond_on_str.append((x,y,'d'))
            if not((x+1, y) in plqlist):      # right plaquette
                bond_on_str.append((x+1,y,'d'))
        
        elif xperiodic == True:
            if x == 0:
                if not((nx-2, y) in plqlist):     # left plaquette (PBC)
                    bond_on_str.append((0,y,'d'))
                if not((x+1, y) in plqlist):      # right plaquette
                    bond_on_str.append((x+1,y,'d'))
            elif x == nx - 2:
                if not((x-1, y) in plqlist):      # left plaquette
                    bond_on_str.append((x,y,'d'))
                if not((0, y) in plqlist):        # right plaquette (PBC)
                    bond_on_str.append((0,y,'d'))
            # dealing with plaquettes off boundary as usual
            else: 
                if not((x-1, y) in plqlist):      # left plaquette
                    bond_on_str.append((x,y,'d'))
                if not((x+1, y) in plqlist):      # right plaquette
                    bond_on_str.append((x+1,y,'d'))

    circum = len(bond_on_str)       
    return bond_on_str, area, circum

def selectRegion(string, width, args):
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
    size = [args['nx'], args['ny']]
    xperiodic = args['xperiodic']
    for bond in string:
        x, y, dir = bond[0], bond[1], bond[2]
        if dir == 'd':
            # add the plaquette to the left/right of this bond
            left = [(x-1,y,'r'),(x-1,y,'d'),(x,y,'d'),(x-1,y+1,'r')]
            right = [(x,y,'d'),(x,y,'r'),(x+1,y,'d'),(x,y+1,'r')]
            # check whether this is a boundary bond
            if xperiodic == False:
                if x == 0:                # left boundary: add right plq. only
                    left = []
                elif x == size[0]  - 1:   # right boundary: add left plq. only
                    right = []
            elif xperiodic == True:
                if x == 0:                # left boundary: add rightmost plq.
                    left = [(x-1,y,'r'),(x-1,y,'d'),(x,y,'d'),(x-1,y+1,'r')]
                elif x == size[0]  - 1:   # right boundary: add leftmost plq.
                    right = [(x,y,'d'),(x,y,'r'),(x+1,y,'d'),(x,y+1,'r')]
            for newbond in left: 
                region.append(lat.lat(newbond[0:2], newbond[2], size, xperiodic))
            for newbond in right:
                region.append(lat.lat(newbond[0:2], newbond[2], size, xperiodic))
        if dir == 'r':
            # add the plaquette above/below this bond
            above = [(x,y-1,'r'),(x,y-1,'d'),(x+1,y-1,'d'),(x,y,'r')]
            below = [(x,y,'r'),(x,y,'d'),(x+1,y,'d'),(x,y+1,'r')]
            if y == 0:              # upper boundary; add plq. below only
                above = []
            if y == size[1] - 1:    # lower boundary; add plq. above only
                below = []                
            for newbond in above: 
                region.append(lat.lat(newbond[0:2], newbond[2], size, xperiodic))
            for newbond in below:
                region.append(lat.lat(newbond[0:2], newbond[2], size, xperiodic))
    region = np.asarray(region, dtype=int)
    region = np.unique(region)
    return region
