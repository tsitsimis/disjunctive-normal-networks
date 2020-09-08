from functools import reduce

from disjunctive_nn import DisjunctiveNormalNetwork
from sklearn import datasets


def test_dummy():
    assert True


def test_fit():
    n_samples = 400
    noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.1)
    X, y = noisy_moons

    n_polytopes = 2
    m = 4
    dnn = DisjunctiveNormalNetwork(n_polytopes=n_polytopes, m=m)
    dnn.fit(X, y, epochs=5000, lr=0.01)

    assert (len(dnn.polytopes) == n_polytopes) \
        and reduce(lambda a, b: a and b, [polytope.m == m for polytope in dnn.polytopes])
