""" Description of Dalitz phase space """

__author__ = "Vitaly Vorobyev"
__email__ = "vit.vorobiev@gmail.com"
__date__ = "January 11, 2019"

import numpy as np

def __RotateEuler__(x, key):
    """ Auxiliary function for random rotation """
    sp, st = np.sin(x['alpha']), np.sin(x['beta'])
    cp, ct = np.cos(x['alpha']), np.cos(x['beta'])
    sk, ck = -sp, cp
    return np.array([
        np.sum(x[key] * np.array([ck*ct*cp - sk*sp, -sk*ct*cp - ck*sp, st*cp]).T, axis=-1),
        np.sum(x[key] * np.array([ck*ct*sp + sk*cp, -sk*ct*sp + ck*cp, st*sp]).T, axis=-1),
        np.sum(x[key] * np.array([-ck*st, sk*st, ct]).T, axis=-1)
    ]).T

def unpack_data(data):
    """ dict to lists and types """
    print(data.dtype)
    rtypes = data.dtype.columns
    return [data[rtypes[0]], data[rtypes[1]], rtypes[0], rtypes[1]]

class DalitzPhaseSpace(object):
    """ Dalitz phase space for spinless particles """
    def __init__(self, ma, mb, mc, md):
        """ Constructor """
        self.area = None
        self.mass = np.array([ma, mb, mc, md])
        self.m = {'A' : ma, 'B' : mb, 'C' : mc, 'M' : md}
        # Lookups for several useful values
        self.mass_sq = self.mass**2
        self.msq = {'A' : ma**2, 'B' : mb**2, 'C' : mc**2, 'M' : md**2}
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
        return (self.m['M'] - self.other_mass[rtype])**2
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
        return (self.msq['M'] - mr_sq - self.other_mass_sq[rtype]) /\
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
        return (self.msq['M'] + mr_sq - self.other_mass_sq[rtype]) / (2. * msq['M'])
    def momentum_res(self, mr_sq, rtype):
        """ Resonance momentum in the D rest frame """
        return np.sqrt(abs(self.energy_res(mr_sq, rtype)**2 - mr_sq))
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
        mask = (msq2 > self.mass_sq_range[rtype2][0]) &\
               (msq2 < self.mass_sq_range[rtype2][1])
        msq1_min, msq1_max = self.mr_sq_range(rtype1, msq2[mask], rtype2)
        mask[mask] = (msq1[mask] >= msq1_min) & (msq1_max >= msq1[mask])
        return mask

    def uniform_sample(self, rtype1, rtype2, nevt, majorant=None, area=False):
        """ Uniformly distributed events """
        msq1_lo, msq1_hi = self.mass_sq_range[rtype1]
        msq2_lo, msq2_hi = self.mass_sq_range[rtype2]
        dtype = [(rtype1, np.float), (rtype2, np.float)]
        data = np.empty(min(10**6, 3*nevt), dtype)
        rect_square = (msq1_hi - msq1_lo) * (msq2_hi - msq2_lo)
        counts, msq1, msq2 = 0, [], []
        if majorant is not None:
            maj = []
        while len(msq1) < nevt:
            new_msq1 = np.random.uniform(msq1_lo, msq1_hi, min(10**6, 3*nevt))
            new_msq2 = np.random.uniform(msq2_lo, msq2_hi, min(10**6, 3*nevt))
            counts += len(new_msq2)
            data[rtype1], data[rtype2] = new_msq1, new_msq2
            mask = self.inside(data)
            msq1 = msq1 + new_msq1[mask].tolist()
            msq2 = msq2 + new_msq2[mask].tolist()
            if majorant is not None:
                add_maj = np.random.uniform(0, majorant, min(10**6, 3*nevt))
                maj = maj + add_maj[mask].tolist()
        if area:
            self.area = rect_square * len(msq1) / counts
            print 'area', self.area
        data = np.empty(nevt, dtype=dtype)
        data[rtype1], data[rtype2] = msq1[:nevt], msq2[:nevt]
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
        """ Finds third comnination """
        return list(set(['AB', 'AC', 'BC']) - set([rt1, rt2]))[0]

    def dalitzToLotentz(self, data):
        """ """
        msq1, msq2, rt1, rt2 = unpack_data(data)
        df = np.empty(len(msq1), dtype=[
            (rt1, np.float), (rt2, np.float),
            (self.thirdCombination(rt1, rt2), np.float),
            ('alpha', np.float), ('beta', np.float),
            ('eA', np.float), ('eB', np.float), ('eC', np.float),
            ('pA', np.float, (3,)), ('pB', np.float, (3,)), ('pC', np.float, (3,))
            ])
        df[rt1], df[rt2], df[self.thirdCombination(rt1, rt2)] = msq1, msq2, self.third_mass_sq(msq1, msq2)
        df['alpha'] = np.random.rand(len(df)) * np.pi * 2
        df['beta'] = np.random.rand(len(df)) * 2. - 1.
        
        df['eA'] = (df['AB'] + df['AC'] - self.msq['B'] - self.msq['C']) / (2. * self.m['M'])
        df['eB'] = (self.msq['M'] + self.msq['B'] - df['AC']) / (2. * self.m['M'])
        df['eC'] = (self.msq['M'] + self.msq['C'] - df['AB']) / (2. * self.m['M'])

        pzA = np.sqrt(df['eA']**2 - self.msq['A'])
        pzB = (self.msq['B'] + self.msq['A'] + 2.*df['eA']*df['eB'] - df['AB']) / (2.*pzA)
        pzC = (self.msq['C'] + self.msq['A'] + 2.*df['eA']*df['eC'] - df['AC']) / (2.*pzA)
        pxB = np.sqrt(np.abs(df['eB']**2 - pzB**2 - self.msq['B']))

        df['pA'] = np.array([np.zeros(len(df)), np.zeros(len(df)), pzA]).T
        df['pB'] = np.array([ pxB,              np.zeros(len(df)), pzB]).T
        df['pC'] = np.array([-pxB,              np.zeros(len(df)), pzC]).T
        for key in ['pA', 'pB', 'pC']:
            df[key] = __RotateEuler__(df, key)

        # assert(np.allclose(df['AB'], (df['eA'] + df['eB'])**2 - np.sum((df['pA'] + df['pB'])**2, axis=1)))
        # assert(np.allclose(df['AC'], (df['eA'] + df['eC'])**2 - np.sum((df['pA'] + df['pC'])**2, axis=1)))
        # assert(np.allclose(df['BC'], (df['eB'] + df['eC'])**2 - np.sum((df['pB'] + df['pC'])**2, axis=1)))
        return df

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
    data = dphsp.uniform_sample('AB', 'BC', 1000000)
    # import matplotlib.pyplot as plt
    # from PlotUtility import plot_ddist
    # msq1, msq2, rtype1, rtype2 = unpack_data(data)
    # plot_ddist(msq1, msq2)
    # plt.show()
    data = dphsp.dalitzToLotentz(data)
    # print(data[:10])

if __name__ == '__main__':
    main()
