"""
This module contains the MetaSyntaxTransformer class, which is used to
transform the Lark parse tree into a MetaSyntaxAST.
"""
import typing
from functools import reduce
from typing import List, Union

from lark import Discard, Token, Transformer
from lark.visitors import _DiscardType

from opengrammar.parser.meta_syntax import (
    LHS,
    RHS,
    Conjunction,
    Disjunction,
    MetaSyntaxAST,
    NonTerminal,
    Operands,
    PostfixCapable,
    PostfixType,
    Rule,
    Terminal,
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

    def SEPARATOR(self, token: Token) -> _DiscardType:
        """
        Discards separators.

        :param token: A separator token.
        """
        return Discard

    def DISJUNCTION_SYMBOL(self, token: Token) -> _DiscardType:
        """
        Discards disjunction symbols.

        :param token: A disjunction symbol token.
        """
        return Discard

    def TERMINAL(self, token: Token) -> Terminal:
        return Terminal(symbol=token[1:-1])

    def terminal(self, items: List[Terminal]) -> Terminal:
        return items[0]

    def NON_TERMINAL(self, token: Token) -> NonTerminal:
        return NonTerminal(symbol=token)

    def ZERO_OR_MORE(self, token: Token) -> str:
        return str(token.value)

    def ONE_OR_MORE(self, token: Token) -> str:
        return str(token.value)

    def OPTIONAL(self, token: Token) -> str:
        return str(token.value)

    def postfix(self, items: List[str]) -> PostfixType:
        operator = items[0]
        if operator == "*":
            return PostfixType.ZERO_OR_MORE
        elif operator == "+":
            return PostfixType.ONE_OR_MORE
        elif operator == "?":
            return PostfixType.OPTIONAL
        raise SyntaxError("Unknown PostfixType Defined in Grammar")

    def non_terminal(
        self, items: Union[List[Union[NonTerminal, PostfixType]], List[NonTerminal]]
    ) -> NonTerminal:
        if len(items) > 1:
            nt = typing.cast(NonTerminal, items[0])
            nt.postfix = typing.cast(PostfixType, items[1])
            return nt
        else:
            nt = typing.cast(NonTerminal, items[0])
            return nt

    def lhs_bracketed(self, items: List[Operands]) -> Operands:
        return items[1:-1][0]

    def lhs_left_non_terminal(self, items: List[Operands]) -> Operands:
        statements = reversed(items)
        return reduce(lambda a, c: Conjunction(c, a), statements)

    def lhs_right_non_terminal(self, items: List[Operands]) -> Operands:
        statements = reversed(items)
        return reduce(lambda a, c: Conjunction(c, a), statements)

    def lhs_conjuncts(self, items: List[Operands]) -> Operands:
        return items[0]

    def lhs_disjuncts(self, items: List[Operands]) -> Operands:
        statements = reversed(items)
        return reduce(lambda a, c: Disjunction(c, a), statements)

    def lhs_expressions(self, items: List[Operands]) -> Operands:
        return items[0]

    def lhs(self, items: List[Operands]) -> LHS:
        rule: Union[NonTerminal, Conjunction, Disjunction] = typing.cast(
            Union[NonTerminal, Conjunction, Disjunction], items[0]
        )
        return LHS(rule=rule)

    def rhs_atoms(self, items: List[Operands]) -> Operands:
        return items[0]

    def rhs_conjuncts(self, items: List[Operands]) -> Operands:
        if len(items) > 1:
            statements = reversed(items)
            return reduce(lambda a, c: Conjunction(c, a), statements)
        else:
            return items[0]

    def rhs_disjuncts(self, items: List[Operands]) -> Operands:
        if len(items) > 1:
            statements = reversed(items)
            return reduce(lambda a, c: Disjunction(c, a), statements)
        else:
            return items[0]

    def rhs_expressions(self, items: List[Operands]) -> Operands:
        return items[0]

    def rhs_bracketed(
        self, items: List[Union[Disjunction, Conjunction, PostfixType]]
    ) -> PostfixCapable:
        atoms: Union[Disjunction, Conjunction] = typing.cast(
            Union[Disjunction, Conjunction], items[1:-2][0]
        )
        pt: PostfixType = typing.cast(PostfixType, items[1:][-1])
        atoms.postfix = pt
        return atoms

    def rhs(self, items: List[Operands]) -> RHS:
        return RHS(rule=items[0])

    def rule(self, items: List[Union[LHS, RHS]]) -> Rule:
        lhs: LHS = typing.cast(LHS, items[0])
        rhs: RHS = typing.cast(RHS, items[0])
        return Rule(lhs=lhs, rhs=rhs)

    def syntax(self, items: List[Rule]) -> MetaSyntaxAST:
        for index, item in enumerate(items):
            item.number = index + 1
        return MetaSyntaxAST(rules=items)
