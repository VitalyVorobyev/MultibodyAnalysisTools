""" Model fitter based on ROOT framework """

__author__ = "Vitaly Vorobyev"
__email__ = "vit.vorobiev@gmail.com"
__date__ = "April 2017"

from array import array
import numpy as np
from ROOT import TVirtualFitter

from scipy.special import gammaln

from DalitzPhaseSpace import unpack_data

def log_poisson(n, mu):
    """ log Poisson distribution """
    return -2. * (n + np.log(mu) - mu - gammaln(n+1))

def show_covariance_matrix(fitter):
    """ Print covariance matrix """
    npars = fitter.GetNumberFreeParameters()
    cvmtx = fitter.GetCovarianceMatrix()
    for i in range(npars):
        for j in range(npars):
            print ''

class MLFit(object):
    """ Perform unbinned max likelihood fit """
    model, data = None, None
    nevt = 0
    pars = []
    def __init__(self, model, params, data, makefit=True):
        """ Constructor """
        MLFit.model = model
        MLFit.data = data
        MLFit.nevt = len(data[list(data)[0]])
        TVirtualFitter.SetDefaultFitter('Minuit')
        self.npars = 0
        MLFit.pars = []
        for resonance, parameters in params.iteritems():
            for param, info in parameters.iteritems():
                pname = '_'.join([resonance, param])
                MLFit.pars.append([pname, resonance, param] + info)
                self.npars += 1
        self.results = {}  # Get fit results and update parameters
        if makefit:
            self.fit()
    def fit(self):
        """ Run MIGRAD """
        fitter = TVirtualFitter.Fitter(0, self.npars)
        for parn, info in enumerate(MLFit.pars):
            pname, _, _, val0, err0, loval, hival = info
            fitter.SetParameter(parn, pname, val0, err0, loval, hival)
        fitter.SetFCN(MLFit.fcn)
        arglist = array('d', 10*[0])  # Auxiliary array for MINUIT parameters
        arglist[0] = 0
        fitter.ExecuteCommand('SET PRING', arglist, 2)
        arglist[0] = 5000  # number of function calls
        arglist[1] = 0.01  # tolerance
        fitter.ExecuteCommand('MIGRAD', arglist, 2)
        # Thanks to Anton Poluektov for that part
        for parn, info in enumerate(MLFit.pars):
            self.results[info[0]] = {
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
        self.results['cov'] = fitter.GetCovarianceMatrix()
        print 'llh {}, status {}, edm {}, errdef {}, nvpar {}, nparx {}'.format(
            maxlh[0], fitstatus, edm[0], errdef[0], nvpar[0], nparx[0])
        return self.results
    @staticmethod
    def fcn(npar, grad, fval, parv, iflag):
        """ The FCN. We have to make this method static because
            TVirtualFitter doesn't line 'self' as the first argument... """
        norm = MLFit.model.integrate(max(10**6, 100*MLFit.nevt))
        for par, val in zip(MLFit.pars, parv):
            MLFit.set_param(par[1], par[2], val)
            print '  par {} : {}'.format(par[0], val)
        fval[0] = -2.*np.log(MLFit.model.density(MLFit.data)).sum() +\
                   2.*np.log(norm)*MLFit.nevt
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

class BinnedMLFit(object):
    """ Perform binned max likelihood fit """
    model, datah = None, None
    nbins = None
    nevt = None
    pars = []
    rtype1, rtype2, msq1, msq2 = [None] * 4
    def __init__(self, model, params, data, nbins, makefit=True):
        """ Constructor """
        BinnedMLFit.model = model
        BinnedMLFit.nbins = nbins
        rtype1, rtype2 = list(data)
        BinnedMLFit.nevt = len(data[rtype1])
        rng = [model.mass_sq_range[rtype1], model.mass_sq_range[rtype2]]
        BinnedMLFit.datah = np.histogram2d(data[rtype1, rtype2], nbins, rng)[0]
        # TVirtualFitter.SetDefaultFitter('Minuit2')
        self.npars = 0
        BinnedMLFit.pars = []
        for resonance, parameters in params.iteritems():
            for param, info in parameters.iteritems():
                pname = "_".join([resonance, param])
                BinnedMLFit.pars.append([pname, resonance, param, info])
                self.npars += 1
        self.results = {}  # Get fit results and update parameters
        if makefit:
            self.fit()
    def fit(self):
        """ Run MIGRAD """
        fitter = TVirtualFitter.Fitter(0, self.npars)
        for parn, info in enumerate(BinnedMLFit.pars):
            pname, inf = info[0], info[-1]
            val0, err0, loval, hival = inf
            fitter.SetParameter(parn, pname, val0, err0, loval, hival)
        fitter.SetFCN(BinnedMLFit.fcn)
        arglist = array('d', 10*[0])  # Auxiliary array for MINUIT parameters
        arglist[0] = 0
        fitter.ExecuteCommand('SET PRING', arglist, 2)
        arglist[0] = 5000  # number of function calls
        arglist[1] = 0.01  # tolerance
        fitter.ExecuteCommand('MIGRAD', arglist, 2)
        # Thanks to Anton Poluektov for that part
        for parn, par_info in enumerate(BinnedMLFit.pars):
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
        rtype1, rtype2 = list(BinnedMLFit.data)
        msq1, msq2 = BinnedMLFit.data[rtype1], MLFit.data[rtype2]
        # norm = BinnedMLFit.model.integrate(100*len(msq1))
        for par, val in zip(BinnedMLFit.pars, parv):
            BinnedMLFit.set_param(par[1], par[2], val)
        grid_dens = BinnedMLFit.model.grid_dens(rtype1, rtype2, BinnedMLFit.nbins)
        fval[0] = -2. * log_poisson(grid_dens, BinnedMLFit.datah).sum()
        # fval[0] += 2. * np.log(norm) * len(msq1)
        print 'llh {}, norm {}'.format(fval[0], norm)
    @staticmethod
    def set_param(rname, par, val):
        """ Set value of the parameter """
        if par == 'mass':
            BinnedMLFit.model.set_mass(rname, val)
        elif par == 'width':
            BinnedMLFit.model.set_width(rname, val)
        elif par == 'ampl':
            BinnedMLFit.model.set_ampl(rname, val)
        elif par == 'phase':
            BinnedMLFit.model.set_phase(rname, val)
