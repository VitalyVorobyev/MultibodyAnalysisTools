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

    def __call__(self, mass_sq):
        """ Calculate for s """
        return 1. / (self.mass_sq - mass_sq - 1j * self.mass * self.width)
    
    def dens(self, mass_sq):
        """ Density """
        amp = self.__call__(mass_sq)
        return amp.real**2 + amp.imag**2

    def __str__(self):
        """ For the print method """
        return "Relativistic Breit-Wigner lineshape:\n" +\
        " mass = " + str(self.mass) + ", width = " + str(self.width)

def bwff(mom, mom0, radius, spin):
    """ Blatt-Weisskopf formfactor """
    if spin == 0:
        return 1.
    zvals = radius*np.array([mom, mom0])
    if spin == 1:
        zvals = 1 + zvals**2
    elif spin == 2:
        zvals = zvals**2
        zvals = 9 + 3*zvals + zvals**2
    elif spin == 3:
        zvals = zvals**2
        zvals = 225 + 45*zvals + 6*zvals**2 + zvals**3
    elif spin == 4:
        zvals = (zvals**2 - 45.*zvals+105.)**2 + 25.*zvals*(2.*zvals - 21)
    return np.sqrt(zvals[1] / zvals[0])

class MassDependentWidth(object):
    """ Mass-dependent width for the Breit-Wigner resonanses """
    def __init__(self, mass, width, spin, momentum):
        """ Constructor """
        self.mass = mass
        self.width = width
        self.power = 2*spin+1
        self.momentum = momentum

    def __call__(self, mass, momentum, ffact):
        """ Get width value """
        return self.width * (momentum / self.momentum)**self.power * self.mass /\
               mass * ffact**2

def mass_dep_width(width, mass0, mass, mom0, mom, spin, ffact):
    """ Mass-dependent width for the Breit-Wigner resonanses """
    return ffact**2 * width * (mom / mom0)**(2*spin+1) * mass0 / mass

class VarWidthRelBW(object):
    """ Relativistic Breit-Wigner lineshape with mass-dependent width """
    def __init__(self, mass, width, spin, init_mom):
        """ Constructor """
        self.mass_sq = mass**2
        self.radius = 0.5
        self.spin = spin
        self.width = MassDependentWidth(mass, width, spin, init_mom)

    def __call__(self, mass_sq, momentum):
        """ Calculate for s """
        mass = np.sqrt(mass_sq)
        ffact = bwff(momentum, self.width.momentum, self.radius, self.spin)
        width = self.width(mass, momentum, ffact)
        return 1. / (-mass_sq + self.mass_sq - 1j * mass * width)

    def set_mass(self, mass, momentum):
        """ Change mass """
        self.mass_sq = mass**2
        self.width.mass = mass
        self.width.momentum = momentum

    def set_width(self, width):
        """ Change width """
        self.width.width = width
