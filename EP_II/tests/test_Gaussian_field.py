import numpy as np
from Gaussian_Field import GaussianField

def test_initialization():

    field = GaussianField(L=100,Ng=128,aref=1.0)

    assert field.L == 100
    assert field.Ng == 128
    assert field.V == 100**3
    assert field.aref == 1.0


def test_P_linear():

    field = GaussianField(L=100,Ng=128,aref=1.0)
    k = np.array([0.0, 0.1, 0.2])
    P = field.P_linear(k)

    assert P.shape == k.shape
    assert P[0] == 0.0
    assert np.all(P[1:] >= 0.0)


def test_normal_modes():

    field = GaussianField(L=100,Ng=8,aref=1.0)
    k = field.normal_modes(field.L,field.V,field.Ng)

    assert k.shape == (8, 8, 8)
    assert k[0, 0, 0] == 0.0
    assert np.all(k >= 0.0)


def test_Gaussian_field():

    field = GaussianField(L=100,Ng=8,aref=1.0)
    delta_x, delta_k = field.Gaussian_field()

    assert delta_x.shape == (8, 8, 8)
    assert delta_k.shape == (8, 8, 8)

    assert np.isrealobj(delta_x)
    assert np.all(np.isfinite(delta_x))


def test_zero_mode():

    field = GaussianField(L=100,Ng=8,aref=1.0)
    delta_x, delta_k = field.Gaussian_field()

    assert delta_k[0, 0, 0] == 0.0

def test_hermitian_symmetry():

    field = GaussianField(L=100,Ng=8,aref=1.0)
    delta_x, delta_k = field.Gaussian_field()

    Ng = field.Ng

    for i in range(Ng):
        for j in range(Ng):
            for l in range(Ng):

                im = (-i) % Ng
                jm = (-j) % Ng
                lm = (-l) % Ng

                assert np.allclose(delta_k[im, jm, lm],np.conj(delta_k[i, j, l]))