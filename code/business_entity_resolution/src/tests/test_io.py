import csv

import numpy as np
import pytest

from ber.io import to_id, to_key, write_id_lists


def test_key_roundtrip():
    ids = ["S1-925783039", "S2-5", "S3-100000000"]
    assert to_id(to_key(ids)) == ids


def test_writer_roundtrip(tmp_path):
    p = tmp_path / "m.tsv"
    write_id_lists(p, "matched_entity_ids", ["S1-1", "S1-2"], to_key(["S1-1", "S1-1"]), to_key(["S3-4", "S2-9"]))
    rows = list(csv.reader(open(p), delimiter="\t"))
    assert rows == [["source1_entity_id", "matched_entity_ids"], ["S1-1", "S2-9,S3-4"], ["S1-2", ""]]


def test_writer_rejects_bad(tmp_path):
    p = tmp_path / "m.tsv"
    with pytest.raises(AssertionError):  # dup inside list
        write_id_lists(p, "x", ["S1-1"], to_key(["S1-1", "S1-1"]), to_key(["S2-9", "S2-9"]))
    with pytest.raises(AssertionError):  # self match
        write_id_lists(p, "x", ["S1-1"], to_key(["S1-1"]), to_key(["S1-2"]))
    with pytest.raises(AssertionError):  # S1 not in order
        write_id_lists(p, "x", ["S1-1"], to_key(["S1-7"]), to_key(["S2-2"]))
    with pytest.raises(AssertionError):  # unknown id
        write_id_lists(p, "x", ["S1-1"], to_key(["S1-1"]), to_key(["S2-2"]), valid_keys=np.array([1]))
