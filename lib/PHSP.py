""" Phase space multibody generator """

from ROOT import TTree, TGenPhaseSpace, TLorentzVector, TFile
import numpy as np

def generate(mother, children, nevt):
    """ """
    pMother = TLorentzVector(0., 0., 0., mother)
    event = TGenPhaseSpace()
    event.SetDecay(pMother, len(children), np.array(children))
    file = TFile('test.root', 'RECREATE')
    tree = TTree('evt', 'evt')
    w, mab, mbc = [np.array(np.empty(1), dtype=np.float) for _ in range(3)]
    tree.Branch('w', w, 'w[1]/D')  # weight
    tree.Branch('mab', mab, 'mab[1]/D')  # Dalitz variable
    tree.Branch('mbc', mbc, 'mbc[1]/D')  # Dalitz variable
    chMomenta = []
    for idx, _ in enumerate(children):
        chMomenta.append(np.empty(4, dtype=np.float))
        name = 'ch{}mom'.format(idx)
        tree.Branch(name, chMomenta[-1], name + '[4]/D')
    for _ in range(nevt):
        w[0] = event.Generate()
        for idx, _ in enumerate(children):
            mom = event.GetDecay(idx)
            chMomenta[idx] = np.array([mom.X(), mom.Y(), mom.Z(), mom.E()])
        pAB = event.GetDecay(0) + event.GetDecay(1)
        pBC = event.GetDecay(1) + event.GetDecay(2)
        mab[0] = pAB.M2()
        mbc[0] = pBC.M2()
        tree.Fill()
    file.Write()
    file.Close()

if __name__ == '__main__':
    generate(1.865, [0.576, 0.135, 0.135], 1000000)
