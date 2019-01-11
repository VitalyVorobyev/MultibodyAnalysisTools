""" Code adopted from EvtGen generator """

import numpy as np
from Rotator import RotateEuler

wtmaxDict = {1 : 1./16, 2 : 1./150,  3 : 1./2, 4 : 1./5}

def EvtPawt(a, b, c):
    temp = (a**2 - (b + c)*(b + c)) * (a**2 - (b - c)*(b - c))
    if temp < 0:
        return 0.0
    return np.sqrt(temp) / (2.*a)

def PhaseSpaceTwoBody(mo, ch, nevt):
    """ """
    moms = np.empty((len(ch), 4))
    # Two body phase space
    energy = (mo**2 + ch[0]**2 - ch[1]**2) / (2. * mo)
    assert(energy > ch[0])
    p3 = np.sqrt(energy**2 - ch[0]**2)
    moms[0] = np.array(energy, 0., 0., p3)
    moms[1] = np.array(mo - energy, 0., 0., -p3)

    # Now rotate four vectors
    angles = np.random.rand((nevt, 2))
    alpha = angles[:, 0] * np.pi * 2.
    beta  = angles[:, 1] * 2. - 1.

    moms[:, 1:] = [RotateEuler(p[1:], alpha, beta, -alpha) for p in moms]
    return moms

def PhaseSpace(mo, ch, nevt):
    """ N body phase space routine.
        Returns four vectors in parent frame.
    """
    if len(ch) == 2:
        return PhaseSpaceTwoBody(mo, ch, nevt)

    pm = np.zeros((5, len(ch)))
    pm[0, 0] = pm[4, 0] = mo
    pm[4, -1] = ch[-1]
    psum = np.sum(ch)

    wtmax = 1./15 if len(ch) not in wtmaxDict else wtmaxDict[len(ch)]
    pmax = mo - psum + ch[-1]
    pmin = 0.0
    for il in np.arange(len(ch) - 2, -1, -1):
        pmax = pmax + ch[il]
        pmin = pmin + ch[il + 1]
        wtmax = wtmax * EvtPawt(pmax, pmin, ch[il])

    rnd = np.empty(len(ch))
    for _ in range(10**6):
        #TODO What is going on here?!
        rnd[0] = 1.0
        for il1 in np.arange(2, len(ch)):
            ran = np.random.random()
            for il2 in np.arange(il1 - 1, 0, -1):
                if ran < rnd[il2 - 1]:
                    rnd[il2] = ran
                    break
                rnd[il2] = rnd[il2 - 1]
        rnd[-1] = 0.0

        wt = 1.0
        for il in np.range(len(ch) - 2, -1, -1):
	        pm[4, il] = pm[4, il+1] + ch[il] + (rnd[il] - rnd[il+1]) * (mo - psum)
	        wt = wt * EvtPawt(pm[4, il], pm[4, il+1], ch[il])
        assert(wt < wtmax)
        if wt > (np.random.random() * wtmax):
            break

    moms = np.zeros((4, len(ch)))
    for il in np.arange(1, len(ch)):
        pa = EvtPawt(pm[4][il - 1], pm[4, il], ch[il - 1])
        costh = np.random.random() * 2. - 1.
        sinth = np.sqrt(1. - costh**2)
        phi = np.random.random() * np.pi * 2.
        moms[1:, il-1] = pa * np.array([sinth * np.cos(phi), sinth * np.sin(phi), costh])
        pm[1:4, il] = -moms[1:, il - 1]
        moms[0, il - 1] = np.sqrt(pa**2 + ch[il - 1]**2)
        pm[0, il] = np.sqrt(pa**2 + pm[4, il]**2)

    moms[:, -1] = pm[:-1, -1]
    for ilr in np.arange(2, len(ch) + 1):
        il = len(ch) + 1 - ilr
        be = pm[:-1, il - 1] / pm[-1, il - 1]
        for i1 in range(il, len(ch) + 1):
            bep = np.dot(be, moms[:, il - 1])
            temp = (moms[0, i1 - 1] + bep) / (be[0] + 1.)
            moms[1:, il - 1] = moms[1:, il - 1] + temp * be[1:]
            moms[0, i1 - 1] = bep            
    return moms
