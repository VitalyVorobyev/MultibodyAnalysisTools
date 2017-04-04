#! /usr/bin/python2
""" Illustration for relativistic Breit-Wigner lineshape """

import sys
sys.path.append("../lib/")

import numpy as np
import matplotlib.pyplot as plt
from DalitzPhaseSpace import *
from RelBreitWigner import *

plt.style.use('seaborn-white')
plt.rc('font', size=26)
plt.rc('text', usetex=True)

def bw_plot(mass, width, sigmas=10, num=None):
    """ Simple Brait-Wigner lineshape """
    kstar = RelBW(mass, width)
    dm = sigmas*width
    mass = np.linspace(max(0, mass - dm), mass + dm, 500)
    dens = abs(kstar(mass**2))
    plt.figure(num=num)
    plt.plot(mass, dens, linestyle='-', color='blue')
    if num is None:
        plt.show()

def width_dep_bw_plot(phsp, rtype, mass, width, spin, sigmas=10, num=None):
    """ BW """
    mom0 = phsp.momentum_res(mass**2, rtype)
    dm = sigmas*width
    res = VarWidthRelBW(mass, width, spin, mom0)
    mass = limited_mass_linspace(mass - dm, mass + dm, 500, phsp, rtype)
    mass_sq = mass**2
    space_factor = phsp.phsp_factor(mass_sq, rtype)
    momt = phsp.momentum_res(mass_sq, rtype)
    densty = abs(res(mass_sq, momt))
    plt.figure(num=num)
    plt.plot(mass, densty * space_factor, linestyle='-', color='red')
    if num is None:
        plt.show()

def rho_omega(phsp, rtype, omega_amp, num=None):
    """ rho-omega interference """
    m_rho = .7717
    w_rho = .1490
    m_omega = .78265
    w_omega = .00849
    mass = np.array([m_rho, m_omega])
    mom_rho, mom_omega = phsp.momentum_res(mass**2, rtype)
    rho = VarWidthRelBW(m_rho, w_rho, 1, mom_rho)
    omega = VarWidthRelBW(m_omega, w_omega, 1, mom_omega)
    dm = 5*w_rho
    mass = limited_mass_linspace(m_rho - dm, m_rho + dm, 1000, phsp, rtype)
    mass_sq = mass**2
    space_factor = phsp.phsp_factor(mass_sq, rtype)
    momt = phsp.momentum_res(mass_sq, rtype)
    densty = abs(rho(mass_sq, momt) + omega_amp*omega(mass_sq, momt))
    plt.figure(num=num)
    plt.plot(mass, densty * space_factor, linestyle='-', color='blue')
    if num is None:
        plt.show()

PHSP = DalitzPhaseSpace(0.475, 0.135, 0.135, 1.865)
# bw_plot(0.8937, 0.0484, 10, 1)
# width_dep_bw_plot(PHSP, 'AB', 0.8937, 0.0484, 1, 10, 1)
# plt.grid()

# bw_plot(0.7717, 0.1490, 5, 2)
width_dep_bw_plot(PHSP, 'BC', 0.7717, 0.1490, 1, 5, 2)
# plt.grid()

# PHSPB = DalitzPhaseSpace(1.865, 0.135, 0.135, 5.279)
# bw_plot(0.7717, 0.1490, 5, 3)
# width_dep_bw_plot(PHSPB, 'BC', 0.7717, 0.1490, 1, 5, 3)
OMAMP = 0.03
OMPHA = 90.
OMAM = OMAMP*complex(np.cos(OMPHA * np.pi / 180.), np.sin(OMPHA * np.pi / 180.))
rho_omega(PHSP, 'BC', OMAM, 2)
OMPHA = 120.
OMAM = OMAMP*complex(np.cos(OMPHA * np.pi / 180.), np.sin(OMPHA * np.pi / 180.))
rho_omega(PHSP, 'BC', OMAM, 2)

plt.grid()
plt.show()
