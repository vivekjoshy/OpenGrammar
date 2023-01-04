import os
from queue import Queue
from typing import List

from lark import Lark
from lark.exceptions import VisitError

from opengrammar.logics.gplif.transformer import GPLIFTransformer

script_directory = os.path.dirname(__file__)
with open(
    os.path.join(script_directory, "grammars/formula.lark"), encoding="utf-8"
) as fp:
    formula_grammar = "".join(fp.readlines())


parser = Lark(grammar=formula_grammar, start="wff", ambiguity="explicit")


class GPLIFFormulaParser:
    def __init__(self, formula: str):
        # Meta
        self.formula = formula

        # Lark Specific
        self.parse_tree = parser.parse(self.formula)
        self.transformer = GPLIFTransformer(self.formula)

        # Syntax Tree
        try:
            self.syntax_tree = self.transformer.transform(self.parse_tree)
        except VisitError as e:
            raise e.orig_exc


class GPLIFModel:
    def __init__(self, parsers: List[GPLIFFormulaParser]):
        self.parsers = parsers

        for parser in self.parsers:
            self._traverse_syntax_tree(parser.syntax_tree)

    def _traverse_syntax_tree(self, tree):
        formulas = Queue()
