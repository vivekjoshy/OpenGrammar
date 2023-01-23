import collections
from collections.abc import MutableSet
from typing import List, Optional, Tuple


class Relation(collections.OrderedDict, MutableSet):
    def __init__(self, sequence: Optional[List[Tuple]] = None, *args, **kwargs):
        if not sequence:
            sequence = []
        self.sequence = dict.fromkeys(sequence)
        self.shape = len(sequence[0]) if sequence else 0

        sequence_list = list(self.sequence.keys())

        self.uniform_type = True
        for item in sequence_list:
            if isinstance(item, tuple):
                self.uniform_type = True
            else:
                self.uniform_type = False
                break

        if not self.uniform_type:
            raise TypeError(
                f"'{sequence.__class__.__name__}' contains sequences that are not tuples."
            )

        self.uniform_shape = True
        for item in sequence_list:
            if self.shape == len(item):
                self.uniform_shape = True
            else:
                self.uniform_shape = False
                break

        if not self.uniform_shape:
            raise TypeError(
                f"'{sequence.__class__.__name__}' contains tuples of different shape."
            )

        super().__init__(self.sequence, *args, **kwargs)

    def update(self, *args, **kwargs):
        if kwargs:
            raise TypeError("update() takes no keyword arguments.")

        for s in args:
            for e in s:
                self.add(e)

    def add(self, elem: Tuple):
        if len(elem) > 0:
            if self.shape > 0:
                if self.shape == len(elem):
                    self.shape = len(elem)
                    self[elem] = None
                else:
                    raise TypeError(
                        f"This structure only accepts ordered tuples of length {self.shape}."
                    )
            else:
                self.shape = len(elem)
                self[elem] = None
        else:
            raise TypeError("Cannot add an element of length 0")

    def discard(self, elem: Tuple):
        self.pop(elem, None)

    def __le__(self, other):
        return all(e in other for e in self)

    def __lt__(self, other):
        return self <= other and self != other

    def __ge__(self, other):
        return all(e in self for e in other)

    def __gt__(self, other):
        return self >= other and self != other

    def __repr__(self):
        return "Relation{" + f"{(', '.join(map(repr, self.keys())))}" + "}"

    def __str__(self):
        return "{" + f"{(', '.join(map(repr, self.keys())))}" + "}"

    difference = property(lambda self: self.__sub__)
    difference_update = property(lambda self: self.__isub__)
    intersection = property(lambda self: self.__and__)
    intersection_update = property(lambda self: self.__iand__)
    issubset = property(lambda self: self.__le__)
    issuperset = property(lambda self: self.__ge__)
    symmetric_difference = property(lambda self: self.__xor__)
    symmetric_difference_update = property(lambda self: self.__ixor__)
    union = property(lambda self: self.__or__)
