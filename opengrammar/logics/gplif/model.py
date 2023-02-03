from queue import Queue
from typing import Dict, List, Tuple, TypeVar, Union

from opengrammar.logics.gplif import (
    BinaryConnective,
    Function,
    GPLIFFormulaParser,
    Name,
    Predicate,
    Quantifier,
    UnaryConnective,
)
from opengrammar.logics.gplif.errors import TranslationError
from opengrammar.mathematics.meta.structures import Relation

T = TypeVar("T")


class GPLIFDomain(dict[T]):
    def __init__(self, *args, **kwargs):
        self.glossary: Dict[Name, str] = dict()
        self.domain: Dict[
            Name, Union[Predicate, UnaryConnective, BinaryConnective, Quantifier, None]
        ] = dict()
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        return self.domain[item]

    def __setitem__(self, key, value):
        self.domain[key] = value
        self.glossary[key] = ""


class GPLIFExtensions(dict[T]):
    def __init__(self, *args, **kwargs):
        self.glossary: Dict[Predicate, str] = dict()
        self.extensions: Dict[Tuple[str, int], Relation] = dict()
        super().__init__(*args, **kwargs)

    def __getitem__(self, item: Predicate):
        return self.extensions[(item.symbol, item.arity)]

    def __setitem__(self, key: Predicate, value: Relation):
        self.extensions[(key.symbol, key.arity)] = value
        self.glossary[key] = ""


class GPLIFRelations(dict[T]):
    def __init__(self, *args, **kwargs):
        self.glossary: Dict[Function, str] = dict()
        self.relations: Dict[Tuple[str, int], Relation] = dict()
        super().__init__(*args, **kwargs)

    def __getitem__(self, item: Predicate):
        return self.relations[(item.symbol, item.arity)]

    def __setitem__(self, key: Function, value: Relation):
        self.relations[(key.symbol, key.arity)] = value
        self.glossary[key] = ""


class GPLIFModel:
    def __init__(self, argument: List[GPLIFFormulaParser]):
        self.argument = argument

        # Model Theoretic Entries
        self.domain = GPLIFDomain()
        self.extensions = GPLIFExtensions()
        self.relations = GPLIFRelations()

        # Populate Entries
        self._traverse_syntax_tree()

    def translate(self):
        # Verify Complete Entries
        if any(map(lambda _: _[1].strip() == "", self.domain.glossary.items())):
            raise TranslationError("Incomplete Glossary for Domain")

        if any(map(lambda _: _[1].strip() == "", self.extensions.glossary.items())):
            raise TranslationError("Incomplete Glossary for Domain")

        if any(map(lambda _: _[1].strip() == "", self.relations.items())):
            raise TranslationError("Incomplete Glossary for Domain")

    def _traverse_syntax_tree(self):
        formulas = Queue()
        for premise in self.argument:
            formulas.put(premise.syntax_tree)

        # Top-Down Traversal
        while not formulas.empty():
            # Get Latest Formula
            current_formula = formulas.get()

            # Check Type of Formula
            if isinstance(current_formula, Predicate):
                for name in current_formula.names:
                    self.domain[name] = None

                self.extensions[current_formula] = Relation()

            elif isinstance(current_formula, UnaryConnective):
                formulas.put(current_formula.clause)
            elif isinstance(current_formula, BinaryConnective):
                formulas.put(current_formula.antecedent)
                formulas.put(current_formula.consequent)
            elif isinstance(current_formula, Quantifier):
                formulas.put(current_formula.clause)
