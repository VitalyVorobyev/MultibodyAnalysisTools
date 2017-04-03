""" Complete description of Dalitz phase space """

__author__ = "Vitaly Vorobyev"
__email__ = "vit.vorobiev@gmail.com"
__date__ = "April 2017"

import numpy as np

class DalitzPhaseSpace(object):
    """ Dalitz phase space for spinless particles """
    def __init__(self, ma, mb, mc, md):
        """ Constructor """
        self.mass = np.array([ma, mb, mc, md])
        self.mass_sq = self.mass**2
        self.prod_mass = {
            'AB' : self.mass[[0, 1]],
            'AC' : self.mass[[0, 2]],
            'BC' : self.mass[[1, 2]],
        }
        self.prod_mass_sq = {
            'AB' : self.mass_sq[[0, 1]],
            'AC' : self.mass_sq[[0, 2]],
            'BC' : self.mass_sq[[1, 2]],
        }
        self.other_mass = {
            'AB' : self.mass[2],
            'AC' : self.mass[1],
            'BC' : self.mass[0],
        }
        self.other_mass_sq = {
            'AB' : self.mass_sq[2],
            'AC' : self.mass_sq[1],
            'BC' : self.mass_sq[0],
        }
        self.mass_sq_range = {
            'AB' : [self.mr_sq_min('AB'), self.mr_sq_max('AB')],
            'AC' : [self.mr_sq_min('AC'), self.mr_sq_max('AC')],
            'BC' : [self.mr_sq_min('BC'), self.mr_sq_max('BC')]
        }
    def __str__(self):
        """ to str """
        return 'Phase space D -> ABC, where\n mD = ' + str(self.mass[-1]) +\
            ', mA = ' + str(self.mass[0]) +\
            ', mB = ' + str(self.mass[1]) +\
            ', mC = ' + str(self.mass[2])
    def mr_sq_range(self, rtype, mr2_sq, r2type):
        """ Get m(rtype)^2 range for a m(r2type)^2 value """
        en_c = self.energy_c(mr2_sq, r2type)  # e(C) for D -> rC
        mass_sq_c = self.other_mass_sq[r2type]  # m(C)^2
        if r2type[0] in rtype:
            en_p = self.energy_a(mr2_sq, r2type)  # e(A) for r -> AB
            mass_sq_p = self.prod_mass_sq[r2type][0]  # m(A)^2
        else:
            en_p = self.energy_b(mr2_sq, r2type)  # e(B) for r -> AB
            mass_sq_p = self.prod_mass_sq[r2type][1]  # m(B)^2
        mo_c = np.sqrt((en_c**2 - mass_sq_c).clip(0))
        mo_p = np.sqrt((en_p**2 - mass_sq_p).clip(0))
        en_sum = en_c + en_p
        mo_sum = mo_c + mo_p
        mo_dif = mo_c - mo_p
        return np.array([(en_sum - mo_sum) * (en_sum + mo_sum),
                         (en_sum - mo_dif) * (en_sum + mo_dif)])
    def mr_sq_max(self, rtype):
        """ Get max value of m(AB)^2 """
        return (self.mass[3] - self.other_mass[rtype])**2
    def mr_sq_min(self, rtype):
        """ Get min value of m(R)^2 """
        return (self.prod_mass[rtype][0] + self.prod_mass[rtype][1])**2
    def energy_a(self, mr_sq, rtype):
        """ Energy of particle A in the resonance frame
            for r -> AB decay """
        masses_sq = self.prod_mass_sq[rtype]
        return (mr_sq + masses_sq[0] - masses_sq[1]) / (2. * np.sqrt(mr_sq))
    def energy_b(self, mr_sq, rtype):
        """ Energy of particle B in the resonance frame
            for r -> AB decay """
        masses_sq = self.prod_mass_sq[rtype]
        return (mr_sq + masses_sq[1] - masses_sq[0]) / (2. * np.sqrt(mr_sq))
    def energy_c(self, mr_sq, rtype):
        """ Energy of particle C in the resonance frame
            for D -> (r -> AB)C decay """
        return (self.mass_sq[-1] - mr_sq - self.other_mass_sq[rtype]) /\
               (2. * np.sqrt(mr_sq))
    def energy_res(self, mr_sq, rtype):
        """ Resonance energy in the D rest frame """
        return (self.mass_sq[-1] + mr_sq - self.other_mass_sq[rtype]) /\
               (2. * self.mass[-1])
    def momentum_res(self, mr_sq, rtype):
        """ Resonance momentum in the D rest frame """
        return np.sqrt(self.energy_res(mr_sq, rtype)**2 - mr_sq)
    def mass_a(self):
        """ Mass of product A """
        return self.mass[0]
    def mass_b(self):
        """ Mass of product B """
        return self.mass[1]
    def mass_c(self):
        """ Mass of product C """
        return self.mass[2]
    def mass_d(self):
        """ Mass of mother particle """
        return self.mass[3]
