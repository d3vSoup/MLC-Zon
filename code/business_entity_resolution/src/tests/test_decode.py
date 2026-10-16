import itertools

import numpy as np

from ber.decode import expected_f
from ber.metric import f05


# brute force over all 2^n truth outcomes
def brute(q, k):
    e = 0.0
    for z in itertools.product([0, 1], repeat=len(q)):
        pr = np.prod([qi if zi else 1 - qi for qi, zi in zip(q, z)])
        e += pr * float(f05(sum(z), k, sum(z[:k])))
    return e


def test_expected_f_matches_brute_force():
    rng = np.random.default_rng(0)
    Q = np.sort(rng.random((4, 5)), axis=1)[:, ::-1]
    E = expected_f(Q)
    for i in range(4):
        for k in range(6):
            assert abs(E[i, k] - brute(Q[i], k)) < 1e-9


def test_graph_top2_and_rank():
    from ber.features.graph import rank_in, top2
    g = np.array([0, 0, 0, 1])
    p = np.array([0.9, 0.5, 0.2, 0.7], np.float32)
    assert np.allclose(top2(g, p, 2)[:3], [0.5, 0.9, 0.9]) and np.isnan(top2(g, p, 2)[3])
    assert list(rank_in(g, p)) == [0, 1, 2, 0]
