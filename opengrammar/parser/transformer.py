from typing import List, Union

from lark import Transformer, Token
from lark.visitors import Discard

from opengrammar.parser.syntax import (
    AST,
    Type,
    Variable,
    Expression,
    Abstraction,
    Application,
    ArrowType,
    Judgement,
    Declaration,
    Statement,
    SimpleType,
)


class SyntaxTransformer(Transformer[Token, AST]):
    """
    This class is used to transform the Lark parse tree into an AST.
    """

    def WS(self, token: Token) -> Discard:
        """
        Discards whitespace.

        :param token: A whitespace token.
        """
        return Discard

    def NEWLINE(self, token: Token) -> Discard:
        """
        Discards newlines.

        :param token: A newline token.
        """
        return Discard

    def SEPARATOR(self, token: Token) -> Discard:
        """
        Discards separators.

        :param token: A separator token.
        """
        return Discard

    def SPACE(self, token: Token) -> Discard:
        """
        Discards spaces.

        :param token: A space token.
        """
        return Discard

    def LEFT_PARENTHESIS(self, token: Token) -> Discard:
        """
        Discards left parentheses.

        :param token: A left parenthesis token.
        """
        return Discard

    def RIGHT_PARENTHESIS(self, token: Token) -> Discard:
        """
        Discards right parentheses.

        :param token: A right parenthesis token.
        """
        return Discard

    def LAMBDA(self, token: Token) -> Discard:
        """
        Discards lambda.

        :param token: A lambda token.
        """
        return Discard

    def DOT(self, token: Token) -> Discard:
        """
        Discards dot.

        :param token: A dot token.
        """
        return Discard

    def TURNSTILE(self, token: Token) -> Discard:
        """
        Discards turnstiles.

        :param token: A turnstile token.
        """
        return Discard

    def TO(self, token: Token) -> Discard:
        """
        Discards "to".

        :param token: A "to" token.
        """
        return Discard

    def VARIABLE(self, token: Token) -> Variable:
        """
        Create a Variable object.

        :param token: A variable token.
        :return: A Variable object.
        """
        return Variable(name=token.value)

    def TYPE(self, token: Token) -> SimpleType:
        """
        Create a Type object.

        :param token: A type token.
        :return: A Type object.
        """
        return SimpleType(name=token.value)

    def type(self, types: List[Type]) -> Type:
        """
        Create a Type object.

        :param types: A list of types.
        :return: A Type object.
        """
        return types[0]

    def arrow_type(self, types: List[Type]) -> ArrowType:
        """
        Create an ArrowType object.

        :param types: A list of types.
        :return: An ArrowType object.
        """
        return ArrowType(antecedent=types[0], consequent=types[1])

    def simple_type(self, types: List[Type]) -> SimpleType:
        """
        Create a SimpleType object.

        :param types: A list of types.
        :return: A SimpleType object.
        """
        return types[0]

    def abstraction(
            self, objects: List[Union[Variable, Type, Expression]]
    ) -> Abstraction:
        """
        Create an Abstraction object.

        :param objects: A list of objects in the abstraction.
        :return: An Abstraction object.
        """
        return Abstraction(variable=objects[0], type=objects[1], expression=objects[2])

    def application(self, expressions: List[Expression]) -> Application:
        """
        Create an Application object.

        :param expressions: A list of expressions.
        :return: An Application object.
        """
        return Application(function=expressions[0], argument=expressions[1])

    def expression(
            self, objects: List[Union[Application, Abstraction, Variable]]
    ) -> Expression:
        """
        Create an Expression object.

        :param objects: A list of objects in the expression.
        :return: An Expression object.
        """
        return objects[0]

    def statement(self, objects: List[Union[Expression, Type]]) -> Statement:
        """
        Create a Statement object.

        :param objects: A list of objects in the statement.
        :return: A Statement object.
        """
        return Statement(expression=objects[0], type=objects[1])

    def declaration(self, tokens: List[Union[Expression, Type]]) -> Declaration:
        """
        Create a Declaration object.

        :param tokens: A list of tokens.
        :return: A Declaration object.
        """
        return Declaration(expression=tokens[0], type=tokens[1])

    def context(self, tokens: List[Declaration]) -> List[Declaration]:
        """
        Create a list of declarations.

        :param tokens: A list of tokens.
        :return: A list of declarations.
        """
        return tokens

    def judgement(self, tokens: List[Union[List[Declaration], Statement]]) -> Judgement:
        """
        Create a list of statements.

        :param tokens: A list of tokens.
        :return: A list of statements.
        """
        if len(tokens) == 1:
            return Judgement(context=[], statement=tokens[0])
        else:
            return Judgement(context=tokens[0], statement=tokens[1])

    def start(self, judgements: List[Judgement]) -> AST:
        return AST(judgements=judgements)
