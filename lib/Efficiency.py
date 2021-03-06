""" Detector reconstruction efficiency simumulator """

import numpy as np

class Detector(object):
    """ """
    def __init__(self, thetacut=10., ptcut=0.05, trkprob=1., momres=0.02):
        """ Constructor args:
            - thetacut (degrees): only particles with polar angele in the range [thetacut, 180 - thetacut] are accepted
            - ptcut (GeV): only particles with transverce momentum larger then ptcur are accepted
            - trkprob: probability of succesfull track reconstruction in the detector (single track detection efficiency)
            - momres: momentum resolution
        """
        self.costhsqcut = np.cos(thetacut * np.pi / 180.)**2
        self.ptsqcut = ptcut**2
        self.trkprob = trkprob
        self.momres = momres

    def effMask(self, data, keys=['pA', 'pB', 'pC']):
        """ Apply detection efficiency to data """
        mask = np.ones(len(data), dtype=np.bool)
        for key in keys:
            ptsq, costhsq = self.__ptsq_and_costhsq(data, key)
            mask = np.logical_and(mask, ptsq > self.ptsqcut, costhsq < self.costhsqcut)
        return mask

    def apply(self, data, mask=None):
        """ """
        if mask is None:
            mask = self.effMask(data)
        data = data[mask]
        psam = np.random.normal(0., self.momres, (len(data), 3, 3))
        for idx, key in enumerate('ABC'):
            mass = np.sqrt(data['e'+key]**2 - np.sum(data['p'+key]**2, axis=-1))
            data['p'+key] = psam[idx, :] * psam[:, idx]
            data['e'+key] = np.sqrt(mass**2 + np.sum(data['p'+key]**2, axis=-1))
        return data

    def __ptsq_and_costhsq(self, data, key):
        """ Calculate transverse momentum """
        momsq = data[key]**2
        ptsq, pzsq = momsq[:, 0] + momsq[:, 1], momsq[:, 2]
        return ptsq, pzsq / ptsq

class Efficiency(object):
    """ Calculate efficiency over the Dalitz phase space """
    def __init__(self):
        """ Constructor """
        pass

    def __call__(self, gen, det, rt1, rt2, size, nevt):
        """ Calculate efficiency map """
        data = gen(rt1, rt2, nevt)
        edges = [gen.phsp.mass_sq_range[x] for x in [rt1, rt2]]
        print(edges)
        h0, _, _ = np.histogram2d(data[rt1], data[rt2], bins=(size, size), range=edges)
        nzMask = (h0 != 0)
        dataDet = data[det.effMask(data, ['pA', 'pB', 'pC'])]
        h1, x, y = np.histogram2d(dataDet[rt1], dataDet[rt2], bins=(size, size), range=edges)
        eff = np.zeros(h0.shape, dtype=np.float64)
        print(eff.shape)
        eff[nzMask] = h1[nzMask].astype(np.float64) / h0[nzMask]
        return (eff, x, y)

def main():
    """ Unit test """
    from PlotUtility import effPlot, show_phase_space, plot_ddist
    from DalitzPhaseSpace import DalitzPhaseSpace
    from PhspGen import PhspGen
    import matplotlib.pyplot as plt

    phsp = DalitzPhaseSpace(0.51, 0.135, 0.135, 1.865)
    gen = PhspGen(phsp)
    det = Detector(10, 0.05)
    eff = Efficiency()
    rt1, rt2 = 'AB', 'AC'
    # data = gen(rt1, rt2, 10**6)
    e, x, y = eff(gen, det, rt1, rt2, 500, 10**7)
    show_phase_space(phsp, rt1, rt2, 1)
    # plot_ddist(data[rt1], data[rt2], 200, 1)
    effPlot(e, x, y, 1)
    plt.show()

if __name__ == '__main__':
    main()
