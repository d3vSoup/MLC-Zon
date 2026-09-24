import pandas as pd

from ber.metric import f05, per_entity


def test_statement_example():
    assert abs(f05(2, 3, 2) - 2.5 / 3.5) < 1e-9
    assert round(float(f05(2, 3, 2)), 3) == 0.714


def test_singleton_empty_is_one():
    assert f05(0, 0, 0) == 1.0


def test_singleton_any_pred_is_zero():
    assert f05(0, 2, 0) == 0.0


def test_nonsingleton_empty_is_zero():
    assert f05(3, 0, 0) == 0.0


def test_nonsingleton_only_wrong_is_zero():
    assert f05(2, 2, 0) == 0.0


def test_per_entity_frames():
    truth = pd.DataFrame({"s1_id": ["a", "a", "b"], "cand_id": ["x", "y", "z"]})
    pred = pd.DataFrame({"s1_id": ["a", "a", "a", "c"], "cand_id": ["x", "y", "q", "w"]})
    r = per_entity(["a", "b", "c", "d"], truth, pred)
    assert abs(r.loc["a", "f"] - 2.5 / 3.5) < 1e-9
    assert r.loc["b", "f"] == 0.0 and r.loc["c", "f"] == 0.0 and r.loc["d", "f"] == 1.0


def test_loss_buckets_sum_to_loss():
    import numpy as np
    from ber.analysis import buckets
    # singleton fp, missed, wrong top1, unreachable, partial hit with fp+fn+blocking miss, perfect
    G = np.array([0, 2, 2, 3, 4, 1])
    g_cand = np.array([0, 2, 2, 0, 3, 1])
    k = np.array([1, 0, 1, 1, 3, 1])
    t = np.array([0, 0, 0, 0, 2, 1])
    L = buckets(k, t, G, g_cand)
    assert L.loc[0, "singleton_fp"] == 1 and L.loc[1, "missed_entity"] == 1
    assert L.loc[2, "wrong_top1"] == 1 and L.loc[3, "blocking_miss"] == 1
    assert L.loc[4, ["blocking_miss", "extra_fp", "extra_fn"]].gt(0).all() and L.loc[5].sum() == 0
