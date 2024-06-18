import sparse_numeric_table as snt
import numpy as np
import copy


def test_append_b_to_a():
    prng = np.random.Generator(np.random.MT19937(seed=1337))

    a = snt.testing.make_example_table(prng=prng, size=100)
    assert len(a["elementary_school"]) == 100
    assert len(a["high_school"]) == 10
    assert len(a["university"]) == 1

    acp = copy.deepcopy(a)

    b = snt.testing.make_example_table(prng=prng, size=100)
    assert len(b["elementary_school"]) == 100
    assert len(b["high_school"]) == 10
    assert len(b["university"]) == 1

    a = snt.append(a, b)
    assert len(a["elementary_school"]) == 200
    assert len(a["high_school"]) == 20
    assert len(a["university"]) == 2

    assert len(b["elementary_school"]) == 100
    assert len(b["high_school"]) == 10
    assert len(b["university"]) == 1

    np.testing.assert_array_equal(
        a["elementary_school"][100:], b["elementary_school"]
    )
    np.testing.assert_array_equal(
        a["elementary_school"][:100], acp["elementary_school"]
    )

    np.testing.assert_array_equal(a["high_school"][10:], b["high_school"])
    np.testing.assert_array_equal(a["high_school"][:10], acp["high_school"])

    np.testing.assert_array_equal(a["university"][1:], b["university"])
    np.testing.assert_array_equal(a["university"][:1], acp["university"])
