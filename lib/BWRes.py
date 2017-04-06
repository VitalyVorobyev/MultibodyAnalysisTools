""" Breit-Wigner resonance """

__author__ = "Vitaly Vorobyev"
__email__ = "vit.vorobiev@gmail.com"
__date__ = "April 2017"

from RelBreitWigner import *
from AngularDistribution import *
from DalitzPhaseSpace import *

def barier_factor(mom0, mom, spin):
    """ Orbital Barier factor """
    return (mom / mom0) ** spin

class BWRes(object):
    """ Breit-Wigner resonance """
    def __init__(self, mass, width, spin, momentum):
        """ Constructor """
        self.prop = VarWidthRelBW(mass, width, spin, momentum)
        self.spin = spin
        self.momentum = momentum
    def __call__(self, mass_sq, momentum, cos_hel, mompq):
        """ Complex amplitude """
        return self.prop(mass_sq, momentum) * ang_dist(cos_hel, mompq, self.spin) *\
               barier_factor(self.momentum, momentum, self.spin)