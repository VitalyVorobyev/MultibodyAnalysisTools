"""  """

import numpy as np
from DalitzModel import DalitzModel
from PhspGen import PhspGen

class DPGen(PhspGen):
    """ """
    def __init__(self, model):
        """ Constructor """
        super(DPGen, self).__init__(model)
        self.majorant = None
        self.model = model

    def assess_majorant(self, ntries=10**6):
        """ Assess majorant with ntries random tries """
        data = super(DPGen, self).__call__(ntries, 'AB', 'AC', False)
        self.majorant = 1.5 * max(self.model.density(data))
        return self.majorant

    def __call__(self, nevt, rt1='AB', rt2='AC', full=False, silent=False):
        """ Get sample with Neuman method """
        if self.majorant is None:
            self.assess_majorant()
        msq1, msq2, ngen, ntry = [], [], 0, 0
        while ngen < nevt:
            print('{} {}'.format(ngen, nevt))
            usmpl, rndm = super(DPGen, self).__call__(nevt, rt1, rt2, False, self.majorant)
            mask = self.model.density(usmpl) > rndm
            msq1 += usmpl[rt1][mask].tolist()
            msq2 += usmpl[rt2][mask].tolist()
            ngen += len(msq1)
            ntry += nevt
            if not silent:
                print('{} events generated'.format(ngen))
        print('Efficiency: {}'.format(float(ngen) / ntry))
        data = np.empty(len(msq1), [(rt1, np.float), (rt2, np.float)])
        data[rt1], data[rt2] = msq1, msq2
        return self.dalitzToLotentz(data) if full else data

    def integrate(self, nevt=10**6):
        """ MC integral """
        if self.area is None:
            data = super(DPGen, self).__call__('AB', 'BC', nevt, False, None, True)
        else:
            data = super(DPGen, self).__call__('AB', 'BC', nevt, False)
        return self.model.density(data).sum() / nevt * self.area

def main():
    """ Unit test """
    # gen = DPGen(0.51, 0.135, 0.135, 1.865)
    # gen.add_bw('rho(770)', .770, .1490, 1, 'BC')
    # gen.add_bw('K*', .89166, 0.0508, 1, 'AB', 0.638*np.exp(1j*np.radians(133.2)))
    # data = gen(10**3, 'AB', 'BC', silent=True)

if __name__ == '__main__':
    main()
        
    # def mcmc_sample(self, nevt, rt1='AB', rt2='AC', alpha=0.1, batch=16):
    #     """ Metropolis-Hastings sampling """
    #     msq1_lo, msq1_hi = self.mass_sq_range[rt1]
    #     msq2_lo, msq2_hi = self.mass_sq_range[rt2]
    #     sigma1 = alpha * (msq1_hi - msq1_lo)
    #     sigma2 = alpha * (msq2_hi - msq2_lo)
    #     print('s1 = {}, s2 = {}'.format(sigma1, sigma2))
    #     pos1, pos2 = self.grid(rt1, rt2, batch)
    #     data = np.empty(len(pos1), [(rt1, np.float), (rt2, np.float)])
    #     batch_size = len(data[rt1])
    #     msq1, msq2 = [], []
    #     density = self.density(data)
    #     iteration = 1
    #     ntries, naccepted = 0, 0
    #     while (len(msq1)*batch_size < nevt) & (iteration < 10**6):
    #         if iteration % 1000 == 0:
    #             print('iteration {}, nevt {} / {}'.format(iteration, len(msq1)*batch_size, nevt))
    #         # generate new positions (candidates)
    #         if iteration % 2:  # update pos2
    #             npos1, npos2 = pos1, pos2 + np.random.normal(0, sigma2, batch_size)
    #         else:
    #             npos1, npos2 = pos1 + np.random.normal(0, sigma1, batch_size), pos2
    #         # remove positions outside the phase space
    #         data = {rt1 : npos1, rt2 : npos2}
    #         inside_mask = self.inside(data)
    #         rejected = sum(~inside_mask)
    #         if iteration % 2:
    #             data[rt2][~inside_mask] = pos2[~inside_mask]
    #         else:
    #             data[rt1][~inside_mask] = pos1[~inside_mask]
    #         ndensity = self.density(data)
    #         acc_mask = ndensity / density > np.random.uniform(0, 1, batch_size)
    #         if iteration % 2:
    #             pos2[acc_mask] = npos2[acc_mask]
    #         else:
    #             pos1[acc_mask] = npos1[acc_mask]
    #         density[acc_mask] = ndensity[acc_mask]
    #         msq1 = msq1 + pos1.tolist()
    #         msq2 = msq2 + pos2.tolist()
    #         naccepted += sum(acc_mask) - rejected
    #         ntries += batch_size
    #         iteration += 1
    #     print('Acceptance rate = {}'.format(float(naccepted) / ntries))
    #     data = np.empty(nevt, [(rt1, np.float), (rt2, np.float)])
    #     data[rt1], data[rt2] = msq1, msq2
    #     return {rt1 : np.array(msq1).flatten(), rt2 : np.array(msq2).flatten()}
