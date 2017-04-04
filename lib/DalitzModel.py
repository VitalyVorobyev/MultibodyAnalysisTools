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
        DalitzPhaseSpace.__init__(ma, mb, mc, md)
        self.rlist = {'AB' : [], 'AC' : [], 'BC' : []}
    def add_res(self, res, rtype):
        """ Add resonance to model """
        self.rlist[rtype].append(res)
    def __call__(self, msq1, msq2, rtype1='AB', rtype2='AC'):
        """ Complex amplitude """
        mab_sq, mac_sq, mbc_sq = self.unravel_masses(msq1, msq2, rtype1, rtype2)
        result = 0. + 1j*0.
        if len(self.rlist['AB']) != 0: # D -> R(AB) + C
            cos_hel = self.cos_hel(mac_sq, 'AC')
            mom_a = self.momentum_a(mab_sq, 'AB')
            mom_r = self.momentum_res(mab_sq, 'AB')
            mompq = mom_a * mom_r
            for res in self.rlist['AB']:
                if isinstance(res, BWRes):
                    result += res(mab_sq, mom_r, cos_hel, mompq)
        if len(self.rlist['AC']) != 0: # D -> R(AC) + B
            cos_hel = self.cos_hel(mab_sq, 'AB')
            mom_a = self.momentum_a(mac_sq, 'AC')
            mom_r = self.momentum_res(mac_sq, 'AC')
            mompq = mom_a * mom_r
            for res in self.rlist['AC']:
                if isinstance(res, BWRes):
                    result += res(mac_sq, mom_r, cos_hel, mompq)
        if len(self.rlist['BC']) != 0: # D -> R(BC) + A
            cos_hel = self.cos_hel(mac_sq, 'AC')
            mom_a = self.momentum_a(mbc_sq, 'BC')
            mom_r = self.momentum_res(mbc_sq, 'BC')
            mompq = mom_a * mom_r
            for res in self.rlist['BC']:
                if isinstance(res, BWRes):
                    result += res(mbc_sq, mom_r, cos_hel, mompq)
        return result
    def density(self, msq1, msq2, rtype1='AB', rtpe2='AC'):
        """ Probability density """
        return abs(self(msq1, msq2, rtype1, rtpe2))
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
