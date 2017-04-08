""" Model fitter based on ROOT framework """

__author__ = "Vitaly Vorobyev"
__email__ = "vit.vorobiev@gmail.com"
__date__ = "April 2017"

from array import array
import numpy as np
from ROOT import TVirtualFitter

class MLFit(object):
    """ Perform unbinned max likelihood fit """
    model, data = None, None
    pars = []
    rtype1, rtype2, msq1, msq2 = [None] * 4
    def __init__(self, model, params, data, makefit=True):
        """ Constructor """
        MLFit.model = model
        MLFit.data = data
        # TVirtualFitter.SetDefaultFitter('Minuit2')
        self.npars = 0
        MLFit.pars = []
        for resonance, parameters in params.iteritems():
            for param, info in parameters.iteritems():
                # info = [val0, err0, lo, hi]
                pname = "_".join([resonance, param])
                MLFit.pars.append([pname, resonance, param, info])
                self.npars += 1
        self.results = {}  # Get fit results and update parameters
        if makefit:
            self.fit()
    def fit(self):
        """ Run MIGRAD """
        fitter = TVirtualFitter.Fitter(0, self.npars)
        for parn, info in enumerate(MLFit.pars):
            pname, inf = info[0], info[-1]
            val0, err0, loval, hival = inf
            fitter.SetParameter(parn, pname, val0, err0, loval, hival)
        fitter.SetFCN(MLFit.fcn)
        arglist = array('d', 10*[0])  # Auxiliary array for MINUIT parameters
        arglist[0] = 0
        fitter.ExecuteCommand('SET PRING', arglist, 2)
        arglist[0] = 5000  # number of function calls
        arglist[1] = 0.01  # tolerance
        fitter.ExecuteCommand('MIGRAD', arglist, 2)
        # Thanks to Anton Poluektov for that part
        for parn, par_info in enumerate(MLFit.pars):
            self.results[par_info[0]] = {
                'val' : fitter.GetParameter(parn),
                'err' : fitter.GetParError(parn)
            }
        # Get status of minimisation and NLL at the minimum
        maxlh, edm, errdef = [array("d", [0.])] * 3
        nvpar, nparx = [array("i", [0])] * 2
        fitstatus = fitter.GetStats(maxlh, edm, errdef, nvpar, nparx)

        # return fit results
        self.results["loglh"] = maxlh[0]
        self.results["status"] = fitstatus
        return self.results
    @staticmethod
    def fcn(npar, grad, fval, parv, iflag):
        """ The FCN """
        rtype1, rtype2 = list(MLFit.data)
        msq1, msq2 = MLFit.data[rtype1], MLFit.data[rtype2]
        norm = MLFit.model.integrate(100*len(msq1))
        for par, val in zip(MLFit.pars, parv):
            MLFit.set_param(par[1], par[2], val)
        fval[0] = -2. * np.log(MLFit.model.density(msq1, msq2, rtype1, rtype2)).sum()
        fval[0] += 2. * np.log(norm) * len(msq1)
        print 'llh {}, norm {}'.format(fval[0], norm)
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
