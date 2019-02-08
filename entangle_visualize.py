# 
#   entangle_visualize.py
#   Toric_Code-Python
#   Visualize virtual bond dimension
#
#   created on Feb 7, 2019 by Yue Zhengyuan
#   code influcenced by iTensor - tevol.h
#

import numpy as np
import matplotlib.pyplot as plt

def visualize(mps, para):
    """
    Visualize virtual bond dimensions of an MPS

    Parameters
    --------------
    mps : list of numpy arrays
        MPS to be visualized
    para : parameter dictionary
        Parameters of the Toric Code lattice
    """
    return mps