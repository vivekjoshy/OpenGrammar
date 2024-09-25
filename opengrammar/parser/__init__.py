"""
The parser module contains the parser for the syntax of OpenGrammar.
"""

from pathlib import Path

import rich
from lark import Lark

from opengrammar.parser.syntax import AST
from opengrammar.parser.transformer import SyntaxTransformer

# Load Syntax
syntax_path = Path(__file__).parent / "syntax.lark"
with open(syntax_path, "r", encoding="utf-8") as _s:
    _syntax = _s.read()


class SyntaxParser:
    """
    Parses Syntax into an AST.
    """

    def __init__(self) -> None:
        """
        Initializes the SyntaxParser.
        """
        self._lark_parser: Lark = Lark(
            grammar=_syntax, start="start", parser="earley", ambiguity="explicit"
        )

    def parse(self, text: str) -> AST:
        """
        Parses the text into an AST.

        :param text: A string with valid Syntax.
        :return: An AST.
        """
        _grammar_tree = self._lark_parser.parse(text)
        _grammar_transformer = SyntaxTransformer()
        _grammar_ast = _grammar_transformer.transform(_grammar_tree)
        return _grammar_ast


if __name__ == "__main__":
    source_code_path = Path(__file__).parent / "proof.og"
    with open(source_code_path, "r", encoding="utf-8") as _s:
        _source_code = _s.read()

    parser = SyntaxParser()
    ast = parser.parse(_source_code)
    rich.print(ast)
