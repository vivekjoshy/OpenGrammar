from abc import ABC
from collections.abc import Hashable, Set


class OrderedSet(Set, Hashable, ABC):
    __hash__ = Set._hash

    wrapped_methods = (
        "difference",
        "difference_update",
        "intersection",
        "intersection_update",
        "issubset",
        "issuperset",
        "symmetric_difference",
        "symmetric_difference_update",
        "union",
        "copy",
    )

    def __repr__(self):
        return f"OrderedSet{set(self._set)}"

    def __new__(cls, iterable=None):
        self_object = super(OrderedSet, cls).__new__(OrderedSet)
        self_object._set = frozenset() if iterable is None else frozenset(iterable)
        for method_name in cls.wrapped_methods:
            setattr(
                self_object, method_name, cls._wrap_method(method_name, self_object)
            )
        return self_object

    @classmethod
    def _wrap_method(cls, method_name, obj):
        def method(*args, **kwargs):
            result = getattr(obj._set, method_name)(*args, **kwargs)
            return OrderedSet(result)

        return method

    def __getattr__(self, attribute):
        return getattr(self._set, attribute)

    def __getitem__(self, index):
        return list(self._set)[index]

    def __contains__(self, item):
        return item in self._set

    def __len__(self):
        return len(self._set)

    def __iter__(self):
        return iter(self._set)
