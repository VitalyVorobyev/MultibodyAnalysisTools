#! /usr/bin/python2
""" Illustration for relativistic Breit-Wigner lineshape """

__author__ = "Vitaly Vorobyev"
__email__ = "vit.vorobiev@gmail.com"
__date__ = "April 2017"

import sys
sys.path.append("../lib/")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from DalitzPhaseSpace import *
from RelBreitWigner import *

plt.style.use('seaborn-white')
plt.rc('font', size=26)
plt.rc('text', usetex=True)

def width_dep_bw(phsp, rtype, mass, width, spin, masses):
    """ BW """
    mom0 = phsp.momentum_res(mass**2, rtype)
    res = VarWidthRelBW(mass, width, spin, mom0)
    mass_sq = masses**2
    momt = phsp.momentum_res(mass_sq, rtype)
    return res(mass_sq, momt)

NFRAMES = 200
OMAMP = 0.03
PHSP = DalitzPhaseSpace(.475, .135, .135, 1.865)
RTYPE = 'BC'
MRHO = .7717
WRHO = .1490
MOMEGA = .78265
WOMEGA = .00849
MASSES = limited_mass_linspace(0, 100, 1000, PHSP, RTYPE)
MASSES_SQ = MASSES**2
RHO_AMP = width_dep_bw(PHSP, RTYPE, MRHO, WRHO, 1, MASSES)
OMEGA_AMP = width_dep_bw(PHSP, RTYPE, MOMEGA, WOMEGA, 1, MASSES)
SPACE_FACTOR = PHSP.phsp_factor(MASSES_SQ, RTYPE)

FIG = plt.figure()
AXES = plt.axes(xlim=(min(MASSES), max(MASSES)), ylim=(0, 1.05*max(abs(RHO_AMP+OMAMP*OMEGA_AMP))))
LINE, = AXES.plot([], [], lw=1, linestyle='-', color='blue')

def init():
    """ Init frame """
    LINE.set_data([], [])
    return LINE,

def animate(i):
    """ Update frame """
    print 'Frame', i, '/', NFRAMES
    ampl_omega = OMAMP*complex(np.cos(i * np.pi / 100), np.sin(i * np.pi / 100))
    LINE.set_data(MASSES, abs(RHO_AMP + ampl_omega * OMEGA_AMP) * SPACE_FACTOR)
    return LINE,

def rho_omega_animated():
    """ Make animation! """
    anim = animation.FuncAnimation(FIG, animate, init_func=init,
                                   frames=NFRAMES, interval=20, blit=True)
    # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    anim.save('basic_animation.gif', dpi=80, writer='imagemagick')

if __name__ == '__main__':
    rho_omega_animated()
