""" Angular distributions """

__author__ = "Vitaly Vorobyev"
__email__ = "vit.vorobiev@gmail.com"
__date__ = "April 2017"

def ang_dist(cos_hel, mompq, spin):
    """ Angular distribution """
    if spin == 0:
        return 1
    if spin == 1:
        return -2. * cos_hel * mompq
    elif spin == 2:
        return 16./3 * (3.*cos_hel**2 - 1) * mompq**2
    elif spin == 3:
        return -8./5 * (5.*cos_hel**3 - 3.*cos_hel) * mompq**3
    elif spin == 4:
        cos_hel_sq = cos_hel**2
        return 16./35 * (35.*cos_hel_sq**2 - 30.*cos_hel_sq + 3) * mompq**4
    elif spin == 5:
        return -32./63 * (63.*cos_hel_sq**5 - 70.*cos_hel**3 + 15.*cos_hel) * mompq**5
    else:
        return 1.
