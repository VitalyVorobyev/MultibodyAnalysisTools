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
        if len(self.rlist['AB']) != 0: # D -> R(AB) + C
            mom_a = self.momentum_a(mab_sq, 'AB')
            mom_c = self.momentum_c(mab_sq, 'AB')
            eng_a = np.sqrt(mom_a**2 + self.mass_sq[0])
            eng_c = np.sqrt(mom_c**2 + self.mass_sq[2])
            cos_hel = (self.mass_sq[0] + self.mass_sq[2] + 2.*eng_a*eng_c - mac_sq) /\
                      (2. * mom_a * mom_c)
            mom_r = self.momentum_res(mab_sq, 'AB')
            mompq = mom_a * mom_r
            for res, ampl in self.rlist['AB']:
                if isinstance(res, BWRes):
                    result += ampl*res(mab_sq, mom_r, cos_hel, mompq)
        if len(self.rlist['AC']) != 0: # D -> R(AC) + B
            mom_c = self.momentum_b(mac_sq, 'AC')
            mom_b = self.momentum_c(mac_sq, 'AC')
            eng_c = np.sqrt(mom_c**2 + self.mass_sq[2])
            eng_b = np.sqrt(mom_b**2 + self.mass_sq[1])
            cos_hel = (self.mass_sq[1] + self.mass_sq[2] + 2.*eng_b*eng_c - mbc_sq) /\
                      (2. * mom_b * mom_c)
            mom_r = self.momentum_res(mac_sq, 'AC')
            mompq = mom_c * mom_r
            for res, ampl in self.rlist['AC']:
                if isinstance(res, BWRes):
                    result += ampl*res(mac_sq, mom_r, cos_hel, mompq)
        if len(self.rlist['BC']) != 0: # D -> R(BC) + A
            mom_b = self.momentum_a(mbc_sq, 'BC')
            mom_a = self.momentum_c(mbc_sq, 'BC')
            eng_b = np.sqrt(mom_b**2 + self.mass_sq[1])
            eng_a = np.sqrt(mom_a**2 + self.mass_sq[0])
            cos_hel = (self.mass_sq[0] + self.mass_sq[1] + 2.*eng_a*eng_b - mab_sq) /\
                      (2. * mom_a * mom_b)
            mom_r = self.momentum_res(mbc_sq, 'BC')
            mompq = mom_b * mom_r
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
