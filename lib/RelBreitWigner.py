""" Implementation of the relativistic Breit-Wigner lineshape """

__author__ = "Vitaly Vorobyev"
__email__ = "vit.vorobiev@gmail.com"
__date__ = "April 2017"

import numpy as np

class RelBW(object):
    """ Relativistic Breit-Wigner lineshape """
    def __init__(self, mass, width):
        """ Constructor """
        self.mass = mass
        self.mass_sq = mass**2
        self.width = width
    def __call__(self, cur_mass_sq):
        """ Calculate for s """
        return 1. / (self.mass_sq - cur_mass_sq - 1j * self.mass * self.width)
    def __str__(self):
        """ For the print method """
        return "Relativistic Breit-Wigner lineshape:\n" +\
        " mass = " + str(self.mass) + ", width = " + str(self.width)

class MassDependentWidth(object):
    """ Mass-dependent width for the Breit-Wigner resonanses """
    def __init__(self, width, spin, momentum):
        """ Constructor """
        self.width = width
        self.power = 2*spin+1
        self.momentum = momentum
    def __call__(self, mass, momentum):
        """ Get width value """
        return self.width * (momentum / self.momentum)**self.power

class VarWidthRelBW(object):
    """ Relativistic Breit-Wigner lineshape with mass-dependent width """
    def __init__(self, mass, width, init_mom, spin):
        """ Constructor """
        self.mass_sq = mass**2
        self.width = MassDependentWidth(width, spin, init_mom)
    def __call__(self, mass_sq, momentum):
        """ Calculate for s """
        mass = np.sqrt(mass_sq)
        return 1. / (self.mass_sq - mass_sq - 1j * mass * self.width(mass, momentum))
