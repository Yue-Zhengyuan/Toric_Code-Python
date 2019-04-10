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
import copy
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
            for i in range(args['nx'] - 1):
                str_pair.append((i, y1, 'r'))
            for i in range(args['nx'] - 1):
                str_pair.append((i, y2, 'r'))
            str_pair_list.append(copy.copy(str_pair))

        return str_pair_list
    
    if args['xperiodic'] == False:

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
            closed_str_list.append(copy.copy(closed_str))
        return closed_str_list
