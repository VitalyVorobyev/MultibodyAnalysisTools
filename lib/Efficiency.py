""" Detector reconstruction efficiency emulator """

import numpy as np

class Detector(object):
    """ """
    def __init__(self, thetacut=10. / 180. * np.pi, trkprob=0.9, ptcut=0.05):
        """ """
        self.thetacut = thetacut
        self.trkprob = trkprob
        self.ptcut = ptcut

    def apply(self, data):
        """ Apply detector model to data """
        
