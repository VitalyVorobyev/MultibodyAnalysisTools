""" Dalitz decay amplitude model """

__author__ = "Vitaly Vorobyev"
__email__ = "vit.vorobiev@gmail.com"
__date__ = "April 2017"

import numpy as np
from DalitzPhaseSpace import *
from BWRes import *

class DalitzModel(DalitzPhaseSpace):
    """ Dalitz model """
    def __init__(self, ma, mb, mc, md):
        """ Constructor with DalitzPhaseSpace object """
        super(DalitzModel, self).__init__(ma, mb, mc, md)
        self.rlist = {'AB' : [], 'AC' : [], 'BC' : []}
        self.rdict = {}

    def add_res(self, name, prop, rtype, ampl=1.+1j*.0):
        """ Add resonance to model
            Args:
             - prop : propagator
             - ampl : complex amplitude
        """
        self.rdict[name] = {'prop' : prop, 'ampl' : ampl, 'type' : rtype}
        self.rlist[rtype].append(name)

    def add_bw(self, name, mass, width, spin, rtype, ampl=1.+1j*.0):
        """ Add Breit-Wigner resonance """
        momentum = self.momentum_res(mass**2, rtype)
        self.add_res(name, BWRes(mass, width, spin, momentum), rtype, ampl)

    def set_mass(self, rname, mass):
        """ Set mass of a resonance """
        momentum = self.momentum_res(mass**2, self.rdict[rname]['type'])
        self.rdict[rname]['prop'].set_mass(mass, momentum)

    def set_width(self, rname, width):
        """ Set width of a resonance """
        self.rdict[rname]['prop'].set_width(width)

    def set_ampl(self, rname, ampl):
        """ Set amplitude of a resonance """
        self.rdict[rname]['ampl'] = ampl * np.exp(1j*np.angle(self.rdict[rname]['ampl']))

    def set_phase(self, rname, phase):
        """ Set phase of a resonance """
        self.rdict[rname]['ampl'] = abs(self.rdict[rname]['ampl']) * np.exp(1j*phase)

    def __call__(self, data):
        """ Complex amplitude """
        mab_sq, mac_sq, mbc_sq = self.__unravel_masses__(data)
        result = 0. + 1j*0.

        if len(self.rlist['AB']) != 0:  # D -> R(AB) + C
            cos_hel, mompq, mom_r = self.cos_hel_pq_pr(mab_sq, mac_sq, 'AB', 'AC')
            for rname in self.rlist['AB']:
                res, ampl = self.rdict[rname]['prop'], self.rdict[rname]['ampl']
                if isinstance(res, BWRes):
                    result += ampl*res(mab_sq, mom_r, cos_hel, mompq)

        if len(self.rlist['AC']) != 0:  # D -> R(AC) + B
            cos_hel, mompq, mom_r = self.cos_hel_pq_pr(mac_sq, mab_sq, 'AC', 'AB')
            for rname in self.rlist['AC']:
                res, ampl = self.rdict[rname]['prop'], self.rdict[rname]['ampl']
                if isinstance(res, BWRes):
                    result += ampl*res(mac_sq, mom_r, cos_hel, mompq)

        if len(self.rlist['BC']) != 0:  # D -> R(BC) + A
            cos_hel, mompq, mom_r = self.cos_hel_pq_pr(mbc_sq, mac_sq, 'BC', 'AC')
            for rname in self.rlist['BC']:
                res, ampl = self.rdict[rname]['prop'], self.rdict[rname]['ampl']
                if isinstance(res, BWRes):
                    result += ampl*res(mbc_sq, mom_r, cos_hel, mompq)
                    
        return result

    def density(self, data):
        """ Probability density """
        amp = self(data)
        if isinstance(amp, (np.ndarray, np.generic)):
            amp[~self.inside(data)] = 0
        return amp.real**2 + amp.imag**2

    def grid_dens(self, rt1, rt2, size=500):
        """ Density values in grid nodes """
        min1, max1 = self.mass_sq_range[rt1]
        min2, max2 = self.mass_sq_range[rt2]
        lsp1 = np.linspace(min1, max1, size)
        lsp2 = np.linspace(min2, max2, size)
        msq1g, msq2g = np.meshgrid(lsp1, lsp2)
        data = np.empty((len(msq1g), len(msq2g)), [(rt1, np.float), (rt2, np.float)])
        data[rt1], data[rt2] = msq1g, msq2g
        mask = self.inside(data)
        dens = self.density(data)
        dens[~mask] = 0
        return [msq1g, msq2g, dens]

    def __unravel_masses__(self, data):
        """ Define mAB, mAC and mBC """
        msq1, msq2, rtype1, rtype2 = DalitzPhaseSpace.unpack_data(data)
        if rtype1 == 'AB':
            mab_sq = msq1
            if rtype2 == 'AC':
                mac_sq = msq2
                mbc_sq = self.thirdMsq(mab_sq, mac_sq)
            else:
                mbc_sq = msq2
                mac_sq = self.thirdMsq(mab_sq, mbc_sq)
        elif rtype1 == 'BC':
            mbc_sq = msq1
            if rtype2 == 'AC':
                mac_sq = msq2
                mab_sq = self.thirdMsq(mbc_sq, mac_sq)
            else:
                mab_sq = msq2
                mac_sq = self.thirdMsq(mbc_sq, mab_sq)
        else:
            mac_sq = msq1
            if rtype2 == 'BC':
                mbc_sq = msq2
                mab_sq = self.thirdMsq(mbc_sq, mac_sq)
            else:
                mab_sq = msq2
                mbc_sq = self.thirdMsq(mac_sq, mab_sq)
        return [mab_sq, mac_sq, mbc_sq]
