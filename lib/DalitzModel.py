""" Dalitz decay amplitude model """

__author__ = "Vitaly Vorobyev"
__email__ = "vit.vorobiev@gmail.com"
__date__ = "April 2017"

from DalitzPhaseSpace import *
from BWRes import *

class DalitzModel(DalitzPhaseSpace):
    """ Dalitz model """
    def __init__(self, ma, mb, mc, md):
        """ Constructor with DalitzPhaseSpace object """
        DalitzPhaseSpace.__init__(self, ma, mb, mc, md)
        self.rlist = {'AB' : [], 'AC' : [], 'BC' : []}
        self.majorant = 0
    def add_res(self, res, rtype, ampl=1.+1j*.0):
        """ Add resonance to model """
        self.rlist[rtype].append([res, ampl])
    def add_bw(self, mass, width, spin, rtype, ampl=1.+1j*.0):
        """ Add Breit-Wigner resonance """
        momentum = self.momentum_res(mass**2, rtype)
        self.rlist[rtype].append([BWRes(mass, width, spin, momentum), ampl])
    def __call__(self, msq1, msq2, rtype1='AB', rtype2='AC'):
        """ Complex amplitude """
        mab_sq, mac_sq, mbc_sq = self.unravel_masses(msq1, msq2, rtype1, rtype2)
        result = 0. + 1j*0.
        if len(self.rlist['AB']) != 0:  # D -> R(AB) + C
            cos_hel, mompq, mom_r = self.cos_hel_pq_pr(mab_sq, mac_sq, 'AB', 'AC')
            for res, ampl in self.rlist['AB']:
                if isinstance(res, BWRes):
                    result += ampl*res(mab_sq, mom_r, cos_hel, mompq)
        if len(self.rlist['AC']) != 0:  # D -> R(AC) + B
            cos_hel, mompq, mom_r = self.cos_hel_pq_pr(mac_sq, mab_sq, 'AC', 'AB')
            for res, ampl in self.rlist['AC']:
                if isinstance(res, BWRes):
                    result += ampl*res(mac_sq, mom_r, cos_hel, mompq)
        if len(self.rlist['BC']) != 0:  # D -> R(BC) + A
            cos_hel, mompq, mom_r = self.cos_hel_pq_pr(mbc_sq, mac_sq, 'BC', 'AC')
            for res, ampl in self.rlist['BC']:
                if isinstance(res, BWRes):
                    result += ampl*res(mbc_sq, mom_r, cos_hel, mompq)
        return result
    def density(self, msq1, msq2, rtype1='AB', rtpe2='AC'):
        """ Probability density """
        amp = self(msq1, msq2, rtype1, rtpe2)
        return amp.real**2 + amp.imag**2
    def assess_majorant(self, ntries=10**6):
        """ Assess majorant with ntries random tries """
        mab_sq, mbc_sq = self.uniform_sample('AB', 'BC', ntries)
        self.majorant = 1.5 * max(self.density(mab_sq, mbc_sq, 'AB', 'BC'))
        return self.majorant
    def sample(self, nevt, rtype1='AB', rtype2='AC'):
        """ Get sample with Neuman method """
        if self.majorant == 0:
            self.majorant = self.assess_majorant()
        msq1, msq2 = np.array([]), np.array([])
        while len(msq1) < nevt:
            new_m1sq, new_m2sq, height = self.uniform_sample(rtype1, rtype2, nevt, self.majorant)
            mask = self.density(new_m1sq, new_m2sq, rtype1, rtype2) > height
            msq1 = np.append(msq1, new_m1sq[mask])
            msq2 = np.append(msq2, new_m2sq[mask])
            print len(msq1), 'events generated'
        return msq1[:nevt], msq2[:nevt]
    def grid_dens(self, rtype1, rtype2, size=500):
        """ Density values in grid nodes """
        min1, max1 = self.mass_sq_range[rtype1]
        min2, max2 = self.mass_sq_range[rtype2]
        lsp1 = np.linspace(min1, max1, size)
        lsp2 = np.linspace(min2, max2, size)
        msq1g, msq2g = np.meshgrid(lsp1, lsp2)
        mask = self.inside(msq1g, msq2g, rtype1, rtype2)
        dens = self.density(msq1g, msq2g, rtype1, rtype2)
        dens[~mask] = 0
        return [msq1g, msq2g, dens]
    def unravel_masses(self, msq1, msq2, rtype1, rtype2):
        """ Define mAB, mAC and mBC """
        if rtype1 == 'AB':
            mab_sq = msq1
            if rtype2 == 'AC':
                mac_sq = msq2
                mbc_sq = self.third_mass_sq(mab_sq, mac_sq)
            else:
                mbc_sq = msq2
                mac_sq = self.third_mass_sq(mab_sq, mbc_sq)
        elif rtype1 == 'BC':
            mbc_sq = msq1
            if rtype2 == 'AC':
                mac_sq = msq2
                mab_sq = self.third_mass_sq(mbc_sq, mac_sq)
            else:
                mab_sq = msq2
                mac_sq = self.third_mass_sq(mbc_sq, mab_sq)
        else:
            mac_sq = msq1
            if rtype2 == 'BC':
                mbc_sq = msq2
                mab_sq = self.third_mass_sq(mbc_sq, mac_sq)
            else:
                mab_sq = msq2
                mbc_sq = self.third_mass_sq(mac_sq, mab_sq)
        return [mab_sq, mac_sq, mbc_sq]
