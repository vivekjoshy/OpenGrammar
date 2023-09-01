"""
The parser module contains the parser for the Meta Syntax and the Universal Grammar.
"""

from pathlib import Path

import rich
from lark import Lark

from opengrammar.parser.meta_syntax import MetaSyntaxAST
from opengrammar.parser.transformer import MetaSyntaxTransformer

# Load Meta Syntax
meta_syntax_path = Path(__file__).parent / "meta_syntax.lark"
with open(meta_syntax_path, "r") as _ms:
    _meta_syntax = _ms.read()


class MetaSyntaxParser:
    """
    Parses Meta Syntax into an AST.
    """

    def __init__(self) -> None:
        """
        Initializes the MetaSyntaxParser.
        """
        self._lark_parser: Lark = Lark(
            grammar=_meta_syntax, start="syntax", parser="earley", ambiguity="explicit"
        )

    def parse(self, text: str) -> MetaSyntaxAST:
        """
        Parses the text into an AST.

        :param text: A string with valid Meta Syntax.
        :return: A Meta Syntax AST.
        """
        _grammar_tree = self._lark_parser.parse(text)
        _grammar_transformer = MetaSyntaxTransformer()
        _grammar_ast = _grammar_transformer.transform(_grammar_tree)
        return _grammar_ast


class UniversalParser:
    """
    Parses a Universal Grammar into an AST.
    """

    def __init__(self, grammar: str):
        """
        Initializes the UniversalParser.

        :param grammar: A valid Universal Grammar string.
        """
        self.grammar: str = grammar

        meta_syntax_parser: MetaSyntaxParser = MetaSyntaxParser()
        self.meta_ast = meta_syntax_parser.parse(text=self.grammar)


if __name__ == "__main__":
    with open("grammar.ug", "r") as f:
        sample_grammar = f.read()

    up = UniversalParser(grammar=sample_grammar)
    rich.print(up.meta_ast)
