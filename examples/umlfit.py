#! /usr/bin/python2
""" An example of unbinned max lh fit of Dalitz model paarameters """

import sys
sys.path.append("../lib/")

import numpy as np

from DalitzModel import *
from DalitzModelFitter import *
from PlotUtility import *
import matplotlib.pyplot as plt

model = DalitzModel(.475, .135, .135, 1.865)
model.add_bw('rho(770)', .770, .1490, 1, 'BC')
model.add_bw('K*', .89166, 0.0508, 1, 'AB', 0.638*np.exp(1j*np.radians(133.2)))

data = model.sample(10**3, 'AB', 'BC', silent=True)

params = {
    'K*' : {
        'ampl' : [.6, .1, .4, 1.0],
        'phase' : [2.325, .1, -np.pi, np.pi],
#         'mass' : [.9, .1, .7, 1.1],
#         'width' : [.05, .01, .04, .06]
    },
#     'rho(770)' : {
#         'mass' : [.8, .1, .5, 1.0],
#         'width' : [.150, .05, .1, .2]
#     }
}

MLFit(model, params, data)
