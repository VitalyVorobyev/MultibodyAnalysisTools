""" """

import numpy as np

def RotateEuler(p, phi, theta, xi):
    """ p is a three-vector or array of three-vectors """
    sp, st, sk = np.sin(phi), np.sin(theta), np.sin(xi)
    cp, ct, ck = np.cos(phi), np.cos(theta), np.cos(xi)

    return np.array([
        np.dot(p, np.array([ck*ct*cp - sk*sp, -sk*ct*cp - ck*sp, st*cp])),
        np.dot(p, np.array([ck*ct*sp + sk*cp, -sk*ct*sp + ck*cp, st*sp])),
        np.dot(p, np.array([-ck*st, sk*st, ct]))
    ])

def RandomRotator(moms):
    """ """
    angles = np.random.rand((len(moms), 2))
    alpha = angles[:, 0] * np.pi * 2.
    beta  = angles[:, 1] * 2. - 1.
    return [RotateEuler(p, alpha, beta, -alpha) for p in moms]

def RandomRotation(moms):
    """ """
    angles = np.random.rand(2)
    alpha = angles[0] * np.pi * 2.
    beta  = angles[1] * 2. - 1.
    return RotateEuler(moms, alpha, beta, -alpha)
