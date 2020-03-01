import kwant
import numpy as np
import scipy.sparse.linalg as sla

import logging


def make_system(a=1, t=1.0, r=10):
    """Make QD system with magnetic field
    Docs: https://kwant-project.org/doc/1/tutorial/spectrum
    """
    lat = kwant.lattice.square(a, norbs=1)
    syst = kwant.Builder()

    def circle(pos):
        (x, y) = pos
        rsq = x ** 2 + y ** 2
        return rsq < r ** 2

    def hopx(site1, site2, B):
        # The magnetic field is controlled by the parameter B
        y = site1.pos[1]
        return -t * np.exp(-1j * B * y)

    syst[lat.shape(circle, (0, 0))] = 4 * t
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopx
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = -t
    return syst.finalized()


def spectrum(syst, B):
    ham_mat = syst.hamiltonian_submatrix(params=dict(B=B), sparse=True)
    return sla.eigsh(ham_mat.tocsc(), k=15, sigma=0, return_eigenvectors=False)


class Model:
    a, t, r = 1, 1, 10

    def __init__(self):
        self.syst = make_system(self.a, self.t, self.r)

    def predict(self, features, names=[]):
        logging.info(f"model features: {features}")
        logging.info(f"model names: {names}")

        return np.array([spectrum(self.syst, B) for B in features])

    def tags(self):
        return {"a": self.a, "t": self.t, "r": self.r}
