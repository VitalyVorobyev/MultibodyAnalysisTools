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
        self.rdict = {}
        self.majorant = 0
    def add_res(self, name, prop, rtype, ampl=1.+1j*.0):
        """ Add resonance to model """
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
    def __call__(self, msq1, msq2, rtype1='AB', rtype2='AC'):
        """ Complex amplitude """
        mab_sq, mac_sq, mbc_sq = self.__unravel_masses__(msq1, msq2, rtype1, rtype2)
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
    def density(self, msq1, msq2, rtype1='AB', rtpe2='AC'):
        """ Probability density """
        amp = self(msq1, msq2, rtype1, rtpe2)
        if isinstance(amp, (np.ndarray, np.generic)):
            amp[~self.inside(msq1, msq2, rtype1, rtpe2)] = 0
        return amp.real**2 + amp.imag**2
    def assess_majorant(self, ntries=10**6):
        """ Assess majorant with ntries random tries """
        usmpl = self.uniform_sample('AB', 'BC', ntries)
        self.majorant = 1.5 * max(self.density(usmpl['AB'], usmpl['BC'], 'AB', 'BC'))
        return self.majorant
    def sample(self, nevt, rtype1='AB', rtype2='AC', silent=False):
        """ Get sample with Neuman method """
        if self.majorant == 0:
            self.majorant = self.assess_majorant()
        msq1, msq2 = np.array([]), np.array([])
        while len(msq1) < nevt:
            usmpl = self.uniform_sample(rtype1, rtype2, nevt, self.majorant)
            mask = self.density(usmpl[rtype1], usmpl[rtype2], rtype1, rtype2) > usmpl['h']
            msq1 = np.append(msq1, usmpl[rtype1][mask])
            msq2 = np.append(msq2, usmpl[rtype2][mask])
            if not silent:
                print len(msq1), 'events generated'
        return {rtype1 : msq1[:nevt], rtype2 : msq2[:nevt]}
    def mcmc_sample(self, nevt, rtype1='AB', rtype2='AC', alpha=0.1, batch=16):
        """ Metropolis-Hastings sampling """
        msq1_lo, msq1_hi = self.mass_sq_range[rtype1]
        msq2_lo, msq2_hi = self.mass_sq_range[rtype2]
        sigma1 = alpha * (msq1_hi - msq1_lo)
        sigma2 = alpha * (msq2_hi - msq2_lo)
        print 's1 = {}, s2 = {}'.format(sigma1, sigma2)
        pos1, pos2 = self.grid(rtype1, rtype2, batch)
        pos1, pos2 = pos1[1:-1], pos2[1:-1]
        batch_size = len(pos1)
        msq1, msq2 = [], []
        density = self.density(pos1, pos2, rtype1, rtype2)
        iteration = 1
        ntries, naccepted = 0, 0
        while (len(msq1)*batch_size < nevt) & (iteration < 10**6):
            if iteration % 1000 == 0:
                print 'iteration {}, nevt {} / {}'.format(iteration, len(msq1)*batch_size, nevt)
            # generate new positions (candidates)
            if iteration % 2:  # update pos2
                npos1, npos2 = pos1, pos2 + np.random.normal(0, sigma2, batch_size)
            else:
                npos1, npos2 = pos1 + np.random.normal(0, sigma1, batch_size), pos2
            # remove positions outside the phase space
            inside_mask = self.inside(npos1, npos2, rtype1, rtype2)
            rejected = sum(~inside_mask)
            if iteration % 2:
                npos2[~inside_mask] = pos2[~inside_mask]
            else:
                npos1[~inside_mask] = pos1[~inside_mask]
            ndensity = self.density(npos1, npos2, rtype1, rtype2)
            acc_mask = ndensity / density > np.random.uniform(0, 1, batch_size)
            if iteration % 2:
                pos2[acc_mask] = npos2[acc_mask]
            else:
                pos1[acc_mask] = npos1[acc_mask]
            density[acc_mask] = ndensity[acc_mask]
            msq1.append(list(pos1))
            msq2.append(list(pos2))
            naccepted += sum(acc_mask) - rejected
            ntries += batch_size
            iteration += 1
        print 'Acceptance rate = {}'.format(float(naccepted) / ntries)
        return {rtype1 : np.array(msq1).flatten(), rtype2 : np.array(msq2).flatten()}
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
    def integrate(self, nevt=10**6):
        """ MC integral """
        if self.area is None:
            data = self.uniform_sample('AB', 'BC', nevt, None, True)
        else:
            data = self.uniform_sample('AB', 'BC', nevt)
        return self.density(data['AB'], data['BC'], 'AB', 'BC').sum() / nevt * self.area
    def __unravel_masses__(self, msq1, msq2, rtype1, rtype2):
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
