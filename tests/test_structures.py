from opengrammar.mathematics.meta.structures import OrderedSet


def test_ordered_set():
    assert list(OrderedSet([2, 2, 3, 4, 4, 5])) == [2, 3, 4, 5]
    assert OrderedSet([1, 2, 2, 3, 4, 4, 5]) == OrderedSet([1, 2, 3, 4, 5])
    assert OrderedSet([2, 2, 3, 4, 4, 5]) != OrderedSet([1, 2, 3, 3])
    assert OrderedSet([0, 1, 1, 2, 3, 3, 4, 5, 5])[3] == 3
    assert len(OrderedSet([0, 1, 1, 2, 3, 3, 4, 5, 5])) == 6

    ordered_set = iter(OrderedSet([1, 2, 3]))
    assert ordered_set.__next__() == 1
    assert ordered_set.__next__() == 2
    assert ordered_set.__next__() == 3

    assert getattr(OrderedSet(), "__contains__").__name__ == "__contains__"
    assert OrderedSet().__getattr__("__contains__").__name__ == "__contains__"
    assert repr(OrderedSet([1, 2])) == "OrderedSet{1, 2}"
    assert OrderedSet._wrap_method(
        method_name="difference", obj=OrderedSet([1, 2, 3, 4])
    )(OrderedSet([2, 3, 4])) == OrderedSet([1])
