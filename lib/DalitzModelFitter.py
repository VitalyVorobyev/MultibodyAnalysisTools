""" Model fitter based on ROOT framework """

__author__ = "Vitaly Vorobyev"
__email__ = "vit.vorobiev@gmail.com"
__date__ = "April 2017"

import array
import numpy as np
from ROOT import TVirtualFitter

class MLFit(object):
    """ Perform unbinned max likelihood fit """
    model = None
    pars = []
    rtype1, rtype2, msq1, msq2 = [None] * 4
    def __init__(self, model, params, data):
        """ Constructor """
        MLFit.model = model
        MLFit.rtype1, MLFit.rtype2, MLFit.msq1, MLFit.msq2 = data
        TVirtualFitter.SetDefaultFitter('Minuit2')
        npars = 0
        for resonance, parameters in params.iteritems():
            for param, info in parameters.iteritems():
                # info = [val0, err0, lo, hi]
                pname = "_".join([info[0], info[1]])
                MLFit.pars.append([pname, resonance, param, info])
                npars += 1
        self.fitter = TVirtualFitter.Fitter(0, npars)
        for parn, info in zip(range(npars), MLFit.pars):
            pname, val0, err0, loval, hival = info
            self.fitter.SetParameter(parn, pname, val0, err0, loval, hival)
        self.fitter.SetFCN(MLFit.fcn)
        self.arglist = array.array('d', 10*[0])  # Auxiliary array for MINUIT parameters
        self.arglist[0] = 0
        self.fitter.ExecuteCommand('SET PRING', self.arglist, 2)
        self.arglist[0] = 5000  # number of function calls
        self.arglist[1] = 0.01  # tolerance
        self.fitter.ExecuteCommand('MIGRAD', self.arglist, 2)
        # Thanks to Anton for that part
        self.results = {}  # Get fit results and update parameters
        for parn, par_info in enumerate(MLFit.pars):
            self.results[par_info[0]] = {
                'val' : self.fitter.GetParameter(parn),
                'err' : self.fitter.GetParError(parn)
            }
        # Get status of minimisation and NLL at the minimum
        maxlh = array.array("d", [0.])
        edm = array.array("d", [0.])
        errdef = array.array("d", [0.])
        nvpar = array.array("i", [0])
        nparx = array.array("i", [0])
        fitstatus = self.fitter.GetStats(maxlh, edm, errdef, nvpar, nparx)

        # return fit results
        self.results["loglh"] = maxlh[0]
        self.results["status"] = fitstatus
    @staticmethod
    def fcn(npar, grad, fval, p, iflag):
        """ The FCN """
        for par, val in zip(MLFit.pars, p):
            MLFit.set_param(par[0], par[1], val)
        fval[0] = -2. * np.log(MLFit.model.density(MLFit.rtype1,\
                            MLFit.rtype2, MLFit.msq1, MLFit.msq2)).sum()
    @staticmethod
    def set_param(rname, par, val):
        """ Set value of the parameter """
        if par == 'mass':
            MLFit.model.set_mass(rname, val)
        elif par == 'width':
            MLFit.model.set_width(rname, val)
        elif par == 'ampl':
            MLFit.model.set_ampl(rname, val)
        elif par == 'phase':
            MLFit.model.set_phase(rname, val)
