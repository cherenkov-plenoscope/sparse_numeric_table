import sparse_numeric_table as snt
import numpy as np
import pytest


def test_intersection_dtypes():
    with pytest.raises(AssertionError):
        snt.logic.intersection([1.1])

    for v in [1, -1]:
        r = snt.logic.intersection([v])
        assert r[0] == v


def test_union_dtypes():
    with pytest.raises(AssertionError):
        snt.logic.union([1.1])

    for v in [1, -1]:
        r = snt.logic.union([v])
        assert r[0] == v


def test_difference_dtypes():
    with pytest.raises(AssertionError):
        snt.logic.difference([1.2], [1])

    with pytest.raises(AssertionError):
        snt.logic.difference([1], [1.2])

    with pytest.raises(AssertionError):
        snt.logic.difference([1.2], [1.2])

    for v in [1, -1]:
        r = snt.logic.difference([2], [v])
        assert r[0] == 2


def test_intersection_empty():
    w = snt.logic.intersection()
    assert w.shape[0] == 0
    assert w.dtype == int

    w = snt.logic.intersection([])
    assert w.shape[0] == 0
    assert w.dtype == int

    w = snt.logic.intersection([], [])
    assert w.shape[0] == 0
    assert w.dtype == int


def test_union_empty():
    w = snt.logic.union()
    assert w.shape[0] == 0
    assert w.dtype == int

    w = snt.logic.union([])
    assert w.shape[0] == 0
    assert w.dtype == int

    w = snt.logic.union([], [])
    assert w.shape[0] == 0
    assert w.dtype == int


def test_difference_empty():
    w = snt.logic.difference([])
    assert w.shape[0] == 0
    assert w.dtype == int

    w = snt.logic.difference([], [])
    assert w.shape[0] == 0
    assert w.dtype == int

    w = snt.logic.difference([], [], [])
    assert w.shape[0] == 0
    assert w.dtype == int


def test_intersection_unique():
    with pytest.raises(AssertionError):
        _ = snt.logic.intersection([1, 1])

    with pytest.raises(AssertionError):
        _ = snt.logic.intersection([1, 2, 3], [1, 1])


def test_union_unique():
    with pytest.raises(AssertionError):
        _ = snt.logic.union([1, 1])

    with pytest.raises(AssertionError):
        _ = snt.logic.union([1, 2, 3], [1, 1])


def test_difference_unique():
    with pytest.raises(AssertionError):
        _ = snt.logic.difference([1, 1])

    with pytest.raises(AssertionError):
        _ = snt.logic.difference([1, 2, 3], [1, 1])


def test_union_case1():
    w = snt.logic.union([1], [2], [3, 4, 5])
    for i in [1, 2, 3, 4, 5]:
        assert i in w


def test_union_case2():
    w = snt.logic.union([1], [2], [3, 4, 5], [], [], [])
    for i in [1, 2, 3, 4, 5]:
        assert i in w


def test_intersection_case1():
    w = snt.logic.intersection([1, 2, 3, 4, 5, 6], [2, 4, 6], [1, 2, 3])
    assert w[0] == 2
    assert w.shape[0] == 1


def test_difference_case1():
    w = snt.logic.difference([1, 2, 3, 4, 5, 6], [2, 4, 6], [1, 2, 3])
    assert w[0] == 5
    assert w.shape[0] == 1


def test_difference_case2():
    w = snt.logic.difference([], [1, 2, 3, 4, 5, 6], [2, 4, 6], [1, 2, 3])
    assert w.shape[0] == 0
