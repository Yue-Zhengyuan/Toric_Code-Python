# 
#   str_create.py
#   Toric_Code-Python
#   create closed string operator automatically from input nx and ny
#   for systems long in y direction
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
    bonds on the string (x, y, dir)
    """
    if maxdy > args['ny'] - 1:
        sys.exit("max y separation larger than system y size")
    elif maxdy < 1:
        sys.exit("max y separation smaller than 1")

    if args['xperiodic'] == True:
        # create string pair list by assigning bond
        str_pair_list = []
        mid_y = int(args['ny'] / 2)
        # initial value
        y1 = mid_y - 1
        y2 = mid_y
        for str_sep in range(1, maxdy + 1):
            if str_sep == 1:
                pass
            elif str_sep % 2 == 0:
                y2 += 1
            elif str_sep % 2 == 1:
                y1 -= 1
            str_pair = []
            for i in range(0, args['nx'] - 1):
                str_pair.append((i, y1, 'r'))
            for i in range(0, args['nx'] - 1):
                str_pair.append((i, y2, 'r'))
            # for j in range(y1, y2):
            #     str_pair.append((0, j, 'd'))
            # for j in range(y1, y2):
            #     str_pair.append((args['nx'] - 2, j, 'd'))
            str_pair_list.append(copy(str_pair))

        return str_pair_list
    
    elif args['xperiodic'] == False:

        # create string pair list by assigning plaquette
        closed_str_list = []
        mid_y = int(args['ny'] / 2)
        mid_x = int(args['nx'] / 2)
        # initial value
        y1 = mid_y - 1
        y2 = mid_y
        for str_sep in range(1, maxdy + 1):
            if str_sep == 1:
                pass
            elif str_sep % 2 == 0:
                y2 += 1
            elif str_sep % 2 == 1:
                y1 -= 1
            closed_str = []
            for i in range(args['nx'] - 1):
                closed_str.append((i, y1, 'r'))
            for i in range(args['nx'] - 1):
                closed_str.append((i, y2, 'r'))
            for j in range(y1, y2):
                closed_str.append((0, j, 'd'))
            for j in range(y1, y2):
                closed_str.append((args['nx'] - 1, j, 'd'))
            closed_str_list.append(copy(closed_str))
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
        for i in range(args['nx'] - 2):
            plqlist.append([i, j])
            closed_str_list.append(copy(plqlist))

        # if str_sep == 1:
        #     pass
        # elif str_sep % 2 == 0:
        #     y2 += 1
        # elif str_sep % 2 == 1:
        #     y1 -= 1
        # for i in range(args['nx'] - 1):
        #     for j in range(y1, y2):
        #         plqlist.append([i, j])
        #     closed_str_list.append(copy(plqlist))
        # plqlist.clear()
    return closed_str_list