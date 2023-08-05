"""
This module contains the MetaSyntaxTransformer class, which is used to
transform the Lark parse tree into a MetaSyntaxAST.
"""

import typing
from typing import List, Union

from lark import Discard, Token, Transformer
from lark.visitors import _DiscardType

from opengrammar.parser.meta_syntax import (
    LHS,
    RHS,
    MetaSyntaxAST,
    NonTerminal,
    Rule,
    Terminal,
    Or,
)


class MetaSyntaxTransformer(Transformer[Token, MetaSyntaxAST]):
    """
    Transforms the Lark parse tree into a MetaSyntaxAST.
    """

    def WS(self, token: Token) -> _DiscardType:
        """
        Discards whitespace.

        :param token: A whitespace token.
        """
        return Discard

    def NEWLINE(self, token: Token) -> _DiscardType:
        """
        Discards newlines.

        :param token: A newline token.
        """
        return Discard

    def separator(self, token: Token) -> _DiscardType:
        """
        Discards separators.

        :param token: A separator token.
        """
        return Discard

    def OR_OPERATOR(self, token: Token) -> _DiscardType:
        """
        Discards OR tokens.

        :param token: An OR token.
        """
        return Discard

    def TERMINAL(self, string: str) -> Terminal:
        """
        Creates a terminal from a string.

        :param string: The string with quotes.
        """
        return Terminal(symbol=string[1:-1])

    def NON_TERMINAL(self, string: str) -> NonTerminal:
        """
        Creates a non-terminal from a string.

        :param string: A string without quotes.
        """
        return NonTerminal(symbol=string)

    def non_terminal(self, children: List[NonTerminal]) -> NonTerminal:
        """
        Returns the first child of the non-terminal.

        :param children: A list of non-terminals.
        """
        return children[0]

    def terminal(self, children: List[Terminal]) -> Terminal:
        """
        Returns the first child of the terminal.

        :param children: A list of terminals.
        """
        return children[0]

    def rule(self, children: List[Union[LHS, RHS, Or]]) -> Rule:
        """
        Creates a rule from an LHS and RHS.

        :param children: A list of LHS and RHS.
        """
        lhs: LHS = typing.cast(LHS, children[0])
        rhs: RHS = typing.cast(RHS, children[1])
        return Rule(lhs=lhs, rhs=rhs)

    def lhs(self, children: List[Union[NonTerminal, Terminal]]) -> LHS:
        """
        Creates an LHS from a list of terminals and non-terminals.

        :param children: A list of terminals and non-terminals.
        """
        for index, rule in enumerate(children):
            rule.number = index + 1
        return LHS(rules=children)

    def rhs(self, children: List[Union[NonTerminal, Terminal]]) -> RHS:
        """
        Creates an RHS from a list of terminals and non-terminals.

        :param children: A list of terminals and non-terminals.
        """
        for index, rule in enumerate(children):
            rule.number = index + 1
        return RHS(rules=children)

    def lines(self, children: List[Rule]) -> MetaSyntaxAST:
        """
        Creates a MetaSyntaxAST from a list of rules.

        :param children: A list of rules.
        """
        for index, rule in enumerate(children):
            rule.number = index + 1
        return MetaSyntaxAST(rules=children)
