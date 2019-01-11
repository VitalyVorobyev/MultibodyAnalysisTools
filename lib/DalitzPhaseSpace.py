""" Complete description of Dalitz phase space """

__author__ = "Vitaly Vorobyev"
__email__ = "vit.vorobiev@gmail.com"
__date__ = "April 2017"

import numpy as np
import pandas as pd
from Rotator import RandomRotation

def unpack_data(data):
    """ dict to lists and types """
    rtypes = list(data)
    return [np.array(data[rtypes[0]]), np.array(data[rtypes[1]]), rtypes[0], rtypes[1]]

class DalitzPhaseSpace(object):
    """ Dalitz phase space for spinless particles """
    def __init__(self, ma, mb, mc, md):
        """ Constructor """
        self.area = None
        self.mass = np.array([ma, mb, mc, md])
        # Lookups for several useful values
        self.mass_sq = self.mass**2
        self.msqdict = {'A' : ma**2, 'B' : mb**2, 'C' : mc**2}
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
        self.mass_range = {
            'AB' : np.sqrt(self.mass_sq_range['AB']),
            'AC' : np.sqrt(self.mass_sq_range['AC']),
            'BC' : np.sqrt(self.mass_sq_range['BC'])
        }
    def set_ma(self, ma):
        """ Change m(A) """
        self.__init__(ma, self.mass[1], self.mass[2], self.mass[3])
    def __str__(self):
        """ to str """
        return 'Phase space D -> ABC, where\n mD = ' + str(self.mass[-1]) +\
            ', mA = ' + str(self.mass[0]) +\
            ', mB = ' + str(self.mass[1]) +\
            ', mC = ' + str(self.mass[2])
    def mr_sq_range(self, rtype, mr2_sq, r2type, savep=False):
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
        if not savep:
            return np.array([(en_sum - mo_sum) * (en_sum + mo_sum),
                             (en_sum - mo_dif) * (en_sum + mo_dif)])
        else:
            return [np.array([(en_sum - mo_sum) * (en_sum + mo_sum),
                              (en_sum - mo_dif) * (en_sum + mo_dif)]),
                    mo_c, mo_p]
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
    def momentum_a(self, mr_sq, rtype):
        """ Momentum of particle A in the resonance frame
            for r -> AB decay """
        return np.sqrt(abs(self.energy_a(mr_sq, rtype)**2 - self.prod_mass_sq[rtype][0]))
    def momentum_b(self, mr_sq, rtype):
        """ Momentum of particle B in the resonance frame
            for r -> AB decay (must be equal momentum_a) """
        return np.sqrt(abs(self.energy_b(mr_sq, rtype)**2 - self.prod_mass_sq[rtype][1]))
    def momentum_c(self, mr_sq, rtype):
        """ Momentum of particle C in the resonance frame
            for D -> (r -> AB)C decay """
        return np.sqrt(abs(self.energy_c(mr_sq, rtype)**2 - self.other_mass_sq[rtype]))
    def energy_res(self, mr_sq, rtype):
        """ Resonance energy in the D rest frame """
        return (self.mass_sq[-1] + mr_sq - self.other_mass_sq[rtype]) /\
               (2. * self.mass[-1])
    def momentum_res(self, mr_sq, rtype):
        """ Resonance momentum in the D rest frame """
        return np.sqrt(abs(self.energy_res(mr_sq, rtype)**2 - mr_sq))
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
    def phsp_factor(self, mass_sq, rtype):
        """ Phase space factor """
        rtype2 = 'AC' if rtype != 'AC' else 'BC'
        mr2_mins, mr2_maxs = self.mr_sq_range(rtype2, mass_sq, rtype)
        space_factor = mr2_maxs - mr2_mins
        return space_factor / max(space_factor)
    def cos_hel(self, mr1_sq, mr2_sq, rtype1, rtype2):
        """ Helicity angle """
        mr2_sq_min, mr2_sq_max = self.mr_sq_range(rtype2, mr1_sq, rtype1)
        return (mr2_sq_min + mr2_sq_max - 2.*mr2_sq) / (mr2_sq_max - mr2_sq_min)
    def mompq(self, mr_sq, rtype):
        """ Product of p(A) and p(D) on the r1 frame """
        return self.momentum_a(mr_sq, rtype) * self.momentum_c(mr_sq, rtype)
    def cos_hel_pq_pr(self, mr1_sq, mr2_sq, rtype1, rtype2):
        """ Calculate effectively all kinematics for a resonance """
        mr2_rng, mom_c, mom_p = self.mr_sq_range(rtype2, mr1_sq, rtype1, True)
        mom_r = self.momentum_res(mr1_sq, rtype1)
        np.seterr(divide='ignore')
        cos_hel = np.divide(mr2_rng[0] + mr2_rng[1] - 2.*mr2_sq,\
                            mr2_rng[1] - mr2_rng[0]+0.0000001)
        return cos_hel, mom_c * mom_p, mom_r

    def third_mass_sq(self, mrsq1, mrsq2):
        """ The third Dalitz variable """
        return -(mrsq1 + mrsq2) + sum(self.mass_sq)

    def inside(self, data):
        """ If point inside physical phase space region """
        msq1, msq2, rtype1, rtype2 = unpack_data(data)
        inside = (msq2 > self.mass_sq_range[rtype2][0]) &\
                 (msq2 < self.mass_sq_range[rtype2][1])
        msq1_min, msq1_max = self.mr_sq_range(rtype1, msq2[inside], rtype2)
        inside[inside] = (msq1[inside] >= msq1_min) & (msq1_max >= msq1[inside])
        return inside
    def uniform_sample(self, rtype1, rtype2, nevt, majorant=None, area=False):
        """ Uniformly distributed events """
        msq1_lo, msq1_hi = self.mass_sq_range[rtype1]
        msq2_lo, msq2_hi = self.mass_sq_range[rtype2]
        rect_square = (msq1_hi - msq1_lo) * (msq2_hi - msq2_lo)
        counts, msq1, msq2 = 0, [], []
        if majorant is not None:
            maj = []
        while len(msq1) < nevt:
            print(len(msq1))
            new_msq1 = np.random.uniform(msq1_lo, msq1_hi, min(10**6, 3*nevt))
            new_msq2 = np.random.uniform(msq2_lo, msq2_hi, min(10**6, 3*nevt))
            print(len(new_msq2))
            counts += len(new_msq2)
            mask = self.inside({rtype1 : new_msq1, rtype2 : new_msq2})
            print(mask)
            msq1 = msq1 + new_msq1[mask].tolist()
            msq2 = msq2 + new_msq2[mask].tolist()
            if majorant is not None:
                add_maj = np.random.uniform(0, majorant, min(10**6, 3*nevt))
                maj = maj + add_maj[mask].tolist()
        if area:
            self.area = rect_square * len(msq1) / counts
            print 'area', self.area
        data = {rtype1 : msq1[:nevt], rtype2 : msq2[:nevt]}
        if majorant is not None:
            return [data, maj[:nevt]]
        else:
            return data
    def grid(self, rtype1, rtype2, size=500):
        """ Get a grid within the phase space """
        min1, max1 = self.mass_sq_range[rtype1]
        min2, max2 = self.mass_sq_range[rtype2]
        dm1 = (max1 - min1) / size
        dm2 = (max2 - min2) / size
        lsp1 = np.linspace(min1+0.5*dm1, max1-0.5*dm1, size)
        lsp2 = np.linspace(min2+0.5*dm2, max2-0.5*dm2, size)
        grid = np.meshgrid(lsp1, lsp2)
        msq1 = np.reshape(grid[0], size**2)
        msq2 = np.reshape(grid[1], size**2)
        mask = self.inside({rtype1 : msq1, rtype2 : msq2})
        return [msq1[mask], msq2[mask]]

    def thirdCombination(self, rt1, rt2):
        """ """
        return list(set(['AB', 'AC', 'BC']) - set([rt1, rt2]))[0]

    def dalitzToLotentz(self, data):
        """ """
        msq1, msq2, rt1, rt2 = unpack_data(data)
        df = pd.DataFrame(data)
        df[self.thirdCombination(rt1, rt2)] = self.third_mass_sq(msq1, msq2)
        print(df.head())
        # rtypes = list(data)
        nevt = len(msq1)
        chA = list(set(rt1) & set(rt2))[0]
        chB = list(set(rt1) - set(chA))[0]
        chC = list(set(rt2) - set(chA))[0]
        print(chA, chB, chC)
        msq1 = np.array(msq1)
        msq2 = np.array(msq2)
        print(type(msq1))
        eA = (msq1 + msq2 - self.msqdict[chB] - self.msqdict[chC]) / (2. * self.mass_d())
        eB = (self.mass_sq[-1] + self.msqdict[chB] - msq2) / (2. * self.mass_d())
        eC = (self.mass_sq[-1] + self.msqdict[chC] - msq1) / (2. * self.mass_d())
        pzA = np.sqrt(eA**2 - self.msqdict[chA])
        pzB = (self.msqdict[chB] + self.msqdict[chA] + 2.*eA*eB - msq1) / (2.*pzA)
        pzC = (self.msqdict[chC] + self.msqdict[chA] + 2.*eA*eC - msq2) / (2.*pzA)
        pxB = np.sqrt(eB**2 - pzB**2 - self.msqdict[chB])
        pxC = -pxB
        lvA, lvB, lvC = [], [], []
        for idx in range(nevt):
            moms = RandomRotation(np.array([
                [0., 0., pzA[idx]],
                [pxB[idx], 0., pzB[idx]],
                [pxC[idx], 0., pzC[idx]]
            ]))
            lvA.append([eA[idx]] + moms[0])
            lvB.append([eB[idx]] + moms[1])
            lvC.append([eC[idx]] + moms[2])
        return pd.DataFrame.from_dict({
            rt1 : msq1, rt2 : msq2,
            'lv{}'.format(chA) : lvA,
            'lv{}'.format(chB) : lvB,
            'lv{}'.format(chC) : lvC
            })

def limited_mass_linspace(mmin, mmax, ndots, phsp, rtype):
    """ Set phase space limits if necessary """
    return np.linspace(max(phsp.mass_range[rtype][0], mmin),
                       min(mmax, phsp.mass_range[rtype][1]), ndots+1)[:-1]

def phsp_edge(phsp, rtype1, rtype2):
    """ Calculate phase space edges """
    mr1_min, mr1_max = phsp.mass_sq_range[rtype1]
    mr1_space = np.linspace(mr1_min, mr1_max, 1000)
    mr2_mins, mr2_maxs = phsp.mr_sq_range(rtype2, mr1_space, rtype1)
    mr1 = np.concatenate([mr1_space, mr1_space[::-1]])
    mr2 = np.concatenate([mr2_mins, mr2_maxs[::-1]])
    return [mr1, mr2]

def main():
    """ Unit test """
    dphsp = DalitzPhaseSpace(0.51, 0.135, 0.135, 1.865)
    data = dphsp.uniform_sample('AB', 'AC', 1000)
    # import matplotlib.pyplot as plt
    # from PlotUtility import plot_ddist
    # msq1, msq2, rtype1, rtype2 = unpack_data(data)
    # plot_ddist(msq1, msq2)
    # plt.show()
    data = dphsp.dalitzToLotentz(data)
    print(data.head())

if __name__ == '__main__':
    main()
