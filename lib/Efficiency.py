""" Detector reconstruction efficiency simumulator """

import numpy as np

class Detector(object):
    """ """
    def __init__(self, thetacut=10., ptcut=0.05, trkprob=1.):
        """ """
        self.costhsqcut = np.cos(thetacut * np.pi / 180.)**2
        self.ptsqcut = ptcut**2
        self.trkprob = trkprob

    def effMask(self, data, keys):
        """ Apply detection efficiency to data """
        mask = np.ones(len(data), dtype=np.bool)
        for key in keys:
            ptsq, costhsq = self.__ptsq_and_costhsq(data, key)
            mask = mask & (ptsq > self.ptsqcut) & (costhsq < self.costhsqcut)
        return mask

    def __ptsq_and_costhsq(self, data, key):
        """ Calculate transverse momentum """
        mom = data[key]
        ptsq, pzsq = mom[0]**2 + mom[1]**2, mom[2]**2
        costhsq = pzsq / ptsq + mom[2]**2
        return ptsq, costhsq

class Efficiency(object):
    """ Calculate efficiency over the Dalitz phase space """
    def __init__(self):
        """ Constructor """

    def run(self, phsp, det, rt1, rt2, size, nevt):
        """ Calculate efficiency map """
        data = phsp.dalitzToLotentz(phsp.uniSample(rt1, rt2, nevt))
        edges = [phsp.mass_sq_range[x] for x in [rt1, rt2]]
        h0, _, _ = np.histogram2d(data[rt1], data[rt2], bins=(size, size), range=edges)
        nzMask = (h0 != 0)
        dataDet = data[det.effMask(data, ['pA', 'pB', 'pC'])]
        h1, x, y = np.histogram2d(dataDet[rt1], dataDet[rt2], bins=(size, size), range=edges)
        eff = h1[nzMask].astype(np.float64) / h0[nzMask]
        return eff, x, y

def main():
    """ Unit test """
    # from PlotUtility import effPlot
    from DalitzPhaseSpace import DalitzPhaseSpace
    # import matplotlib.pyplot as plt

    phsp = DalitzPhaseSpace(0.51, 0.135, 0.135, 1.865)
    det = Detector(10, 0.05)
    eff = Efficiency()
    e, x, y = eff.run(phsp, det, 'AB', 'AC', 500, 10**5)
    # effPlot(e, x, y)
    # plt.show()

if __name__ == '__main__':
    main()
