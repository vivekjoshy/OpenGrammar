from queue import Queue
from typing import Dict, List, Tuple, Union

from opengrammar.logics.gplif import (
    BinaryConnective,
    Function,
    GPLIFFormulaParser,
    Name,
    Predicate,
    Quantifier,
    UnaryConnective,
)
from opengrammar.mathematics.meta.structures import Relation


class GPLIFDomain:
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


class GPLIFExtensions(dict):
    def __init__(self, *args, **kwargs):
        self.glossary: Dict[Predicate, str] = dict()
        self.extensions: Dict[Tuple[str, int], Relation] = dict()
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        return self.extensions[item]

    def __setitem__(self, key, value):
        self.extensions[key] = value


class GPLIFRelations(dict):
    def __init__(self, *args, **kwargs):
        self.glossary: Dict[Function, str] = dict()
        self.relations: Dict[Function, Relation] = dict()
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        return self.relations[item]

    def __setitem__(self, key, value):
        self.relations[key] = value


class GPLIFModel:
    def __init__(self, argument: List[GPLIFFormulaParser]):
        self.argument = argument

        # Model Theoretic
        self.domain = GPLIFDomain()
        self.extensions = GPLIFExtensions()
        self.relations: Dict[Function, Relation] = dict()

        # Translation Cache
        self.untranslated_domain = set()
        self.untranslated_extensions = set()
        self.untranslated_relations = set()

    def translate(self):
        pass

    def _prepare(self):
        pass

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
                pass
            elif isinstance(current_formula, UnaryConnective):
                formulas.put(current_formula.clause)
            elif isinstance(current_formula, BinaryConnective):
                formulas.put(current_formula.antecedent)
                formulas.put(current_formula.consequent)
            elif isinstance(current_formula, Quantifier):
                formulas.put(current_formula.clause)
