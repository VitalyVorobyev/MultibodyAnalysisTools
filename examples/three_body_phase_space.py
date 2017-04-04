#! /usr/bin/python2
""" Show kinematically allowed Dlitz plot region """

import sys
sys.path.append("../lib/")

import numpy as np
import matplotlib.pyplot as plt
from DalitzPhaseSpace import DalitzPhaseSpace

plt.style.use('seaborn-white')
plt.rc('font', size=26)
plt.rc('text', usetex=True)

def phsp_edge(phsp, rtype1, rtype2):
    """ Calculate phase space edges """
    mr1_min, mr1_max = phsp.mass_sq_range[rtype1]
    mr1_space = np.linspace(mr1_min, mr1_max, 1000)
    mr2_mins, mr2_maxs = phsp.mr_sq_range(rtype2, mr1_space, rtype1)
    mr1 = np.concatenate([mr1_space, mr1_space[::-1]])
    mr2 = np.concatenate([mr2_mins, mr2_maxs[::-1]])
    return [mr1, mr2]

def show_phase_space(phsp, rtype1, rtype2, num=None):
    """ Plot phase space edge """
    mr1, mr2 = phsp_edge(phsp, rtype1, rtype2)
    mr1_range = max(mr1) - min(mr1)
    mr2_range = max(mr2) - min(mr2)
    height = int(7. * mr1_range / mr2_range)+1 if mr1_range < mr2_range else\
             int(7. * mr2_range / mr1_range)+1
    plt.figure(num=num, figsize=(7, height))
    plt.plot(mr1, mr2, linestyle='-', color='blue')
    plt.gca().set_xlabel(r'$m^{2}_{\mathrm{' + rtype1 + r'}}\ (GeV^{2}/c^{4})$')
    plt.gca().set_ylabel(r'$m^{2}_{\mathrm{' + rtype2 + r'}}\ (GeV^{2}/c^{4})$')
    plt.axis('equal')
    plt.tight_layout()
    plt.xlim(0, 1.05*max(mr1))
    plt.ylim(0, 1.05*max(mr2))
    plt.grid()
    if num is None:
        plt.show()

PHSP = DalitzPhaseSpace(0.475, 0.135, 0.135, 1.865)
print PHSP
mab, mac = PHSP.uniform_sample('AB', 'AC', 1000)
plt.figure()
plt.scatter(mab, mac)
# show_phase_space(PHSP, 'AB', 'AC', 1)
# show_phase_space(PHSP, 'AB', 'BC', 2)
plt.show()
