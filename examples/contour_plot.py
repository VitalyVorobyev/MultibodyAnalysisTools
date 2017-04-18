#! /usr/bin/python2
""" Contour plot for a simple D0 -> Ks0 pi+ pi- decay amplitude model """

__author__ = "Vitaly Vorobyev"
__email__ = "vit.vorobiev@gmail.com"
__date__ = "April 2017"

import sys
sys.path.append("../lib/")

import numpy as np
import matplotlib.pyplot as plt

from DalitzModel import DalitzModel
from PlotUtility import *

# Construct a model
model = DalitzModel(.475, .135, .135, 1.865)
model.add_bw('rho(770)', .770, .1490, 1, 'BC')
model.add_bw('K*', .89166, 0.0508, 1, 'AB', 0.638*np.exp(1j*np.radians(133.2)))

fig = plt.figure(num=12, figsize=(8, 5))
show_phase_space(model, 'AB', 'BC', num=12)
plot_density_countours(model, 'AB', 'BC', num=12)
plt.show()
