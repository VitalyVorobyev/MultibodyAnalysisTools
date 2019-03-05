""" Description of Dalitz phase space """

__author__ = "Vitaly Vorobyev"
__email__ = "vit.vorobiev@gmail.com"
__date__ = "January 11, 2019"

import numpy as np

class DalitzPhaseSpace(object):
    """ Dalitz phase space """
    def __init__(self, ma, mb, mc, md):
        """ Constructor """
        mass = np.array([ma, mb, mc, md])
        mass_sq = mass**2
        self.msqSum = sum(mass_sq)
        # Lookup dicts
        self.m   = {key : val for key, val in zip('ABCM', mass)}
        self.msq = {key : val for key, val in zip('ABCM', mass_sq)}
        self.prod_mass = {
            'AB' : mass[[0, 1]],
            'AC' : mass[[0, 2]],
            'BC' : mass[[1, 2]]}
        self.prod_mass_sq = {
            'AB' : mass_sq[[0, 1]],
            'AC' : mass_sq[[0, 2]],
            'BC' : mass_sq[[1, 2]]}
        self.other_mass = {
            'AB' : mass[2],
            'AC' : mass[1],
            'BC' : mass[0]}
        self.other_mass_sq = {
            'AB' : mass_sq[2],
            'AC' : mass_sq[1],
            'BC' : mass_sq[0]}
        self.mass_sq_range = {
            'AB' : [self.mr_sq_min('AB'), self.mr_sq_max('AB')],
            'AC' : [self.mr_sq_min('AC'), self.mr_sq_max('AC')],
            'BC' : [self.mr_sq_min('BC'), self.mr_sq_max('BC')]}

    def __str__(self):
        """ to str """
        return 'Phase space M -> ABC, where\n  m(M) = {}, m(A) = {}, m(B) = {}, m(C) = {}'\
                .format(self.m['M'], self.m['A'], self.m['B'], self.m['C'])
    @staticmethod
    def unpack_data(data):
        """ dict to lists and types """
        rts = data.dtype.names
        return [data[rts[0]], data[rts[1]], rts[0], rts[1]]

    def mr_sq_range(self, rt, mr2_sq, rt2, savep=False):
        """ Get m(rt)^2 range for a m(r2type)^2 value """
        en_c = self.energy_c(mr2_sq, rt2)  # e(C) for M -> rC
        msq_c = self.other_mass_sq[rt2]  # m(C)^2
        if rt2[0] in rt:
            en_p = self.energy_a(mr2_sq, rt2)  # e(A) for r -> AB
            msq_p = self.prod_mass_sq[rt2][0]  # m(A)^2
        else:
            en_p = self.energy_b(mr2_sq, rt2)  # e(B) for r -> AB
            msq_p = self.prod_mass_sq[rt2][1]  # m(B)^2
        fcn = lambda e, msq: np.sqrt((e**2 - msq).clip(0))
        mo_c, mo_p = fcn(en_c, msq_c), fcn(en_p, msq_p)
        en_sum = en_c + en_p
        mo_sum = mo_c + mo_p
        mo_dif = mo_c - mo_p
        fcn = lambda es, ms: (es - ms) * (es + ms)
        return ([fcn(en_sum, mo_sum), fcn(en_sum, mo_dif)], mo_c, mo_p) if savep else\
                [fcn(en_sum, mo_sum), fcn(en_sum, mo_dif)]

    def mr_sq_max(self, rt):
        """ Get max value of m(AB)^2 """
        return (self.m['M'] - self.other_mass[rt])**2

    def mr_sq_min(self, rt):
        """ Get min value of m(R)^2 """
        return (self.prod_mass[rt][0] + self.prod_mass[rt][1])**2

    def __energy(self, a, b, c, d):
        """ Template for energy calculation """
        return (a + b - c) / (2. * d)

    def __momentum(self, e, msq):
        """ Template for momentum calculation """
        return np.sqrt((e**2 - msq).clip(0))

    def energy_a(self, mr_sq, rt):
        """ Energy of particle A in the resonance frame
            for r -> AB decay """
        msq1, msq2 = self.prod_mass_sq[rt]
        return self.__energy(mr_sq, msq1, msq2, np.sqrt(mr_sq))

    def energy_b(self, mr_sq, rt):
        """ Energy of particle B in the resonance frame
            for r -> AB decay """
        msq1, msq2 = self.prod_mass_sq[rt]
        return self.__energy(mr_sq, msq2, msq1, np.sqrt(mr_sq))

    def energy_c(self, mr_sq, rt):
        """ Energy of particle C in the resonance frame
            for D -> (r -> AB)C decay """
        return self.__energy(self.msq['M'], -mr_sq, self.other_mass_sq[rt], np.sqrt(mr_sq))

    def energy_res(self, mr_sq, rt):
        """ Resonance energy in the D rest frame """
        return self.__energy(self.msq['M'], mr_sq, self.other_mass_sq[rt], self.m['M'])

    def momentum_a(self, mr_sq, rt):
        """ Momentum of particle A in the resonance frame
            for r -> AB decay """
        return self.__momentum(self.energy_a(mr_sq, rt), self.prod_mass_sq[rt][0])

    def momentum_b(self, mr_sq, rt):
        """ Momentum of particle B in the resonance frame
            for r -> AB decay (must be equal momentum_a) """
        return self.__momentum(self.energy_b(mr_sq, rt), self.prod_mass_sq[rt][1])

    def momentum_c(self, mr_sq, rt):
        """ Momentum of particle C in the resonance frame
            for D -> (r -> AB)C decay """
        return self.__momentum(self.energy_c(mr_sq, rt), self.other_mass_sq[rt])

    def momentum_res(self, mr_sq, rt):
        """ Resonance momentum in the D rest frame """
        return self.__momentum(self.energy_res(mr_sq, rt), mr_sq)

    def phsp_factor(self, mass_sq, rt):
        #TODO wtf???
        """ Phase space factor """
        rt2 = 'AC' if rt != 'AC' else 'BC'
        mr2_mins, mr2_maxs = self.mr_sq_range(rt2, mass_sq, rt)
        space_factor = mr2_maxs - mr2_mins
        return space_factor / max(space_factor)

    def cos_hel(self, mr1_sq, mr2_sq, rt1, rt2):
        """ Helicity angle """
        mr2_sq_min, mr2_sq_max = self.mr_sq_range(rt2, mr1_sq, rt1)
        return (mr2_sq_min + mr2_sq_max - 2.*mr2_sq) / (mr2_sq_max - mr2_sq_min)

    def mompq(self, mr_sq, rt):
        """ Product of p(A) and p(D) on the r1 frame """
        return self.momentum_a(mr_sq, rt) * self.momentum_c(mr_sq, rt)

    def cos_hel_pq_pr(self, mr1_sq, mr2_sq, rt1, rt2):
        """ Calculate effectively all kinematics for a resonance """
        mr2_rng, mom_c, mom_p = self.mr_sq_range(rt2, mr1_sq, rt1, True)
        mom_r = self.momentum_res(mr1_sq, rt1)
        np.seterr(divide='ignore')
        cos_hel = np.divide(mr2_rng[0] + mr2_rng[1] - 2.*mr2_sq,\
                            mr2_rng[1] - mr2_rng[0]+0.0000001)
        return cos_hel, mom_c * mom_p, mom_r

    def thirdMsq(self, mrsq1, mrsq2):
        """ The third Dalitz variable """
        return self.msqSum - mrsq1 - mrsq2

    def inside(self, data):
        """ If point inside physical phase space region """
        msq1, msq2, rt1, rt2 = DalitzPhaseSpace.unpack_data(data)
        mask = np.logical_and(msq2 > self.mass_sq_range[rt2][0],
                              msq2 < self.mass_sq_range[rt2][1])
        msq1_min, msq1_max = self.mr_sq_range(rt1, msq2[mask], rt2)
        mask[mask] = np.logical_and(msq1[mask] >= msq1_min,
                                    msq1[mask] <= msq1_max)
        return mask

    def msqLinSp(self, rt, size):
        """ Linspace with 'size' elements along the 'rt' dimension """
        mmin, mmax = self.mass_sq_range[rt]
        dm = (mmax - mmin) / size
        return np.linspace(mmin+0.5*dm, mmax-0.5*dm, size)

    def grid(self, rt1, rt2, size=500):
        """ Makes grid within the phase space """
        data = np.empty(size**2, [(rt1, np.float), (rt2, np.float)])
        grid = np.meshgrid(self.msqLinSp(rt1, size), self.msqLinSp(rt2, size))
        data[rt1] = np.reshape(grid[0], size**2)
        data[rt2] = np.reshape(grid[1], size**2)
        return data[self.inside(data)]

    def thirdRt(self, rt1, rt2):
        """ Finds third combination """
        return list(set(['AB', 'AC', 'BC']) - set([rt1, rt2]))[0]
    __repr__ = __str__

def limited_mass_linspace(mmin, mmax, ndots, phsp, rt):
    """ Set phase space limits if necessary """
    return np.linspace(max(phsp.mass_range[rt][0], mmin),
                       min(mmax, phsp.mass_range[rt][1]), ndots+1)[:-1]

def phsp_edge(phsp, rt1, rt2):
    """ Calculate phase space edges """
    mr1_min, mr1_max = phsp.mass_sq_range[rt1]
    mr1_space = np.linspace(mr1_min, mr1_max, 1000)
    mr2_mins, mr2_maxs = phsp.mr_sq_range(rt2, mr1_space, rt1)
    mr1 = np.concatenate([mr1_space, mr1_space[::-1]])
    mr2 = np.concatenate([mr2_mins, mr2_maxs[::-1]])
    return [mr1, mr2]

def main():
    """ Unit test """
    dphsp = DalitzPhaseSpace(0.51, 0.135, 0.135, 1.865)

if __name__ == '__main__':
    main()
