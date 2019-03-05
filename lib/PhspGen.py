""" Phase space generator for three-body decays """

import numpy as np
from DalitzPhaseSpace import DalitzPhaseSpace

class PhspGen(object):
    """ Phase space generator for three-body decays """
    def __init__(self, phsp):
        """ Constructor """
        self.phsp = phsp
        super(PhspGen, self).__init__(phsp.m['A'], phsp.m['B'], phsp.m['C'], phsp.m['M'])

    def __call__(self, rt1, rt2, nevt, full=True, majorant=None, area=False):
        """ Uniformly distributed events """
        # batch size
        ngen = min(10**6, 3*nevt)
        msq1_lo, msq1_hi = self.phsp.mass_sq_range[rt1]
        msq2_lo, msq2_hi = self.phsp.mass_sq_range[rt2]
        dtype = [(rt1, np.float), (rt2, np.float)]
        data = np.empty(ngen, dtype)
        rect_square = (msq1_hi - msq1_lo) * (msq2_hi - msq2_lo)
        counts, msq1, msq2 = 0, [], []
        if majorant is not None:
            maj = []
        while len(msq1) < nevt:
            new_msq1 = np.random.uniform(msq1_lo, msq1_hi, ngen)
            new_msq2 = np.random.uniform(msq2_lo, msq2_hi, ngen)
            counts += len(new_msq2)
            data[rt1], data[rt2] = new_msq1, new_msq2
            mask = self.phsp.inside(data)
            msq1 = msq1 + new_msq1[mask].tolist()
            msq2 = msq2 + new_msq2[mask].tolist()
            if majorant is not None:
                add_maj = np.random.uniform(0, majorant, ngen)
                maj = maj + add_maj[mask].tolist()
        if area:
            self.phsp.area = rect_square * len(msq1) / counts
            print('area {}'.format(self.phsp.area))
        data = np.empty(nevt, dtype=dtype)
        data[rt1], data[rt2] = msq1[:nevt], msq2[:nevt]
        if full:
            data = self.dalitzToLotentz(data)
        return data if majorant is None else [data, np.array(maj[:nevt])]

    def dalitzToLotentz(self, data):
        """ Derives final-state-particle's momenta from Dalitz variables.
            Adds a random rotation for each event
        """
        msq1, msq2, rt1, rt2 = DalitzPhaseSpace.unpack_data(data)
        df = np.empty(len(msq1), dtype=[
            (rt1, np.float), (rt2, np.float),
            (self.phsp.thirdRt(rt1, rt2), np.float),
            ('alpha', np.float), ('beta', np.float),
            ('eA', np.float), ('eB', np.float), ('eC', np.float),
            ('pA', np.float, (3,)), ('pB', np.float, (3,)), ('pC', np.float, (3,))
            ])
        df[rt1], df[rt2], df[self.phsp.thirdRt(rt1, rt2)] = msq1, msq2, self.phsp.thirdMsq(msq1, msq2)
        df['alpha'] = np.random.rand(len(df)) * 2. * np.pi
        df['beta']  = np.random.rand(len(df)) * 2. - 1.
        
        df['eA'] = (df['AB'] + df['AC'] - self.phsp.msq['B'] - self.phsp.msq['C']) / (2. * self.phsp.m['M'])
        df['eB'] = (self.phsp.msq['M'] + self.phsp.msq['B'] - df['AC']) / (2. * self.phsp.m['M'])
        df['eC'] = (self.phsp.msq['M'] + self.phsp.msq['C'] - df['AB']) / (2. * self.phsp.m['M'])

        pzA = np.sqrt(df['eA']**2 - self.phsp.msq['A'])
        pzB = (self.phsp.msq['B'] + self.phsp.msq['A'] + 2.*df['eA']*df['eB'] - df['AB']) / (2.*pzA)
        pzC = (self.phsp.msq['C'] + self.phsp.msq['A'] + 2.*df['eA']*df['eC'] - df['AC']) / (2.*pzA)
        pxB = np.sqrt(np.abs(df['eB']**2 - pzB**2 - self.msq['B']))

        df['pA'] = np.array([np.zeros(len(df)), np.zeros(len(df)), pzA]).T
        df['pB'] = np.array([              pxB, np.zeros(len(df)), pzB]).T
        df['pC'] = np.array([             -pxB, np.zeros(len(df)), pzC]).T
        for key in ['pA', 'pB', 'pC']:
            df[key] = self.__RotateEuler__(df, key)

        # assert(np.allclose(df['AB'], (df['eA'] + df['eB'])**2 - np.sum((df['pA'] + df['pB'])**2, axis=1)))
        # assert(np.allclose(df['AC'], (df['eA'] + df['eC'])**2 - np.sum((df['pA'] + df['pC'])**2, axis=1)))
        # assert(np.allclose(df['BC'], (df['eB'] + df['eC'])**2 - np.sum((df['pB'] + df['pC'])**2, axis=1)))
        return df

    def __RotateEuler__(self, x, key):
        """ Auxiliary function for random rotation """
        sp, st = np.sin(x['alpha']), np.sin(x['beta'])
        cp, ct = np.cos(x['alpha']), np.cos(x['beta'])
        sk, ck = -sp, cp
        return np.array([
            np.sum(x[key] * np.array([ck*ct*cp - sk*sp, -sk*ct*cp - ck*sp, st*cp]).T, axis=-1),
            np.sum(x[key] * np.array([ck*ct*sp + sk*cp, -sk*ct*sp + ck*cp, st*sp]).T, axis=-1),
            np.sum(x[key] * np.array([-ck*st, sk*st, ct]).T, axis=-1)
        ]).T

def main():
    """ Unit test """
    gen = PhspGen(0.51, 0.135, 0.135, 1.865)
    data = gen('AB', 'BC', 1000000)
    data = gen.dalitzToLotentz(data)

if __name__ == '__main__':
    main()
