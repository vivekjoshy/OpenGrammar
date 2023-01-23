import pytest

from opengrammar.mathematics.meta.structures import Relation


def test_relation():
    r = Relation([(1, 2), (1, 3), (1, 2), (1, 2), (1, 1)])
    assert list(r) == [(1, 2), (1, 3), (1, 1)]

    r.add((2, 1))
    assert list(r) == [(1, 2), (1, 3), (1, 1), (2, 1)]

    r.discard((1, 2))
    assert list(r) == [(1, 3), (1, 1), (2, 1)]
    assert repr(r) == "Relation{(1, 3), (1, 1), (2, 1)}"
    assert str(r) == "{(1, 3), (1, 1), (2, 1)}"

    r.update([(1, 2)])
    assert list(r) == [(1, 3), (1, 1), (2, 1), (1, 2)]

    s = Relation([(1, 1), (1, 2), (1, 3)])
    assert s == s
    assert s != r
    assert s < r
    assert not s > r
    assert s <= r
    assert not s >= r

    with pytest.raises(TypeError):
        Relation([(1, 1), (1, 2), (1, 2, 3)])

    with pytest.raises(TypeError):
        Relation([(1, 1), (1, 2), [1, 2]])

    with pytest.raises(TypeError):
        Relation([(1, 1), (1, 2), 1])

    with pytest.raises(TypeError):
        r.update(a=(1, 2))

    with pytest.raises(TypeError):
        r.add((1, 2, 3))

    with pytest.raises(TypeError):
        r.add(tuple())

    r = Relation()
    r.add((1, 1, 1))
    r.add((1, 1, 2))
    assert str(r) == "{(1, 1, 1), (1, 1, 2)}"
