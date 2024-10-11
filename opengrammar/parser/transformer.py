from typing import List, Union

from lark import Token, Transformer
from lark.visitors import Discard

from opengrammar.parser.syntax import (
    AST,
    Atom,
    BiConditional,
    Conditional,
    Conjunction,
    Declaration,
    Disjunction,
    Equality,
    Expression,
    Judgement,
    Name,
    Negation,
    Predicate,
    QuantifiedExpression,
    QuantifierOperator,
    SimpleType,
    Statement,
    Term,
    Type,
    TypeBiConditional,
    TypeConditional,
    TypeConjunction,
    TypeDisjunction,
    TypeStatement,
    Variable,
)


class SyntaxTransformer(Transformer[Token, AST]):
    def start(self, judgements: List[Judgement]) -> AST:
        return AST(judgements=judgements)

    def judgement(self, items: List[Union[List[Declaration], Statement]]) -> Judgement:
        context = items[0] if len(items) == 2 else []
        statement = items[-1]
        return Judgement(context=context, statement=statement)

    def context(self, declarations: List[Declaration]) -> List[Declaration]:
        return declarations

    def declaration(self, items: List[Declaration]) -> Declaration:
        return items[0]

    def term_declaration(self, items: List[Union[Term, Type]]) -> Declaration:
        return Declaration(term=items[0], type_expression=items[1])

    def type_declaration(self, items: List[Type]) -> Declaration:
        return Declaration(term=items[0], type_expression=items[1])

    def statement(self, expression: Expression) -> Statement:
        return Statement(expression=expression)

    def biconditional(self, items: List[Expression]) -> Expression:
        return BiConditional(antecedent=items[0], consequent=items[1])

    def conditional(self, items: List[Expression]) -> Expression:
        return Conditional(antecedent=items[0], consequent=items[1])

    def disjunction(self, items: List[Expression]) -> Expression:
        return Disjunction(antecedent=items[0], consequent=items[1])

    def conjunction(self, items: List[Expression]) -> Expression:
        return Conjunction(antecedent=items[0], consequent=items[1])

    def negation(self, item: Expression) -> Expression:
        return Negation(expression=item)

    def atom(self, item: Atom) -> Atom:
        return item

    def predicate(self, items: List[Union[str, List[Term]]]) -> Predicate:
        return Predicate(name=items[0], terms=items[1:])

    def quantified(
        self, items: List[Union[QuantifierOperator, Variable, Type, Expression]]
    ) -> QuantifiedExpression:
        return QuantifiedExpression(
            quantifier=items[0],
            variable=items[1],
            type_expression=items[2],
            expression=items[3],
        )

    def equality(self, items: List[Term]) -> Equality:
        return Equality(antecedent=items[0], consequent=items[1])

    def type_expression(self, type_item: Type) -> Type:
        return type_item

    def type_biconditional(self, items: List[Type]) -> Type:
        return TypeBiConditional(antecedent=items[0], consequent=items[1])

    def type_conditional(self, items: List[Type]) -> Type:
        return TypeConditional(antecedent=items[0], consequent=items[1])

    def type_disjunction(self, items: List[Type]) -> Type:
        return TypeDisjunction(antecedent=items[0], consequent=items[1])

    def type_conjunction(self, items: List[Type]) -> Type:
        return TypeConjunction(antecedent=items[0], consequent=items[1])

    def type_atom(self, items: List[Type]) -> Type:
        return items[0]

    def type_statement(self, items: List[Union[Term, Type]]) -> TypeStatement:
        return TypeStatement(term=items[0], type_expression=items[1])

    def term(self, item: Term) -> Term:
        return item

    # Terminal Transformations
    def TYPE(self, token: Token) -> SimpleType:
        return SimpleType(name=token.value)

    def NAME(self, token: Token) -> Name:
        return Name(name=token.value)

    def VARIABLE(self, token: Token) -> Variable:
        return Variable(name=token.value)

    def PREDICATE_NAME(self, token: Token) -> str:
        return token.value

    def FORALL(self, _) -> QuantifierOperator:
        return QuantifierOperator.UNIVERSAL

    def EXISTS(self, _) -> QuantifierOperator:
        return QuantifierOperator.EXISTENTIAL

    # Discard Tokens
    def TURNSTILE(self, _):
        return Discard

    def SEPARATOR(self, _):
        return Discard

    def LEFT_PARENTHESIS(self, _):
        return Discard

    def RIGHT_PARENTHESIS(self, _):
        return Discard

    def COMMA(self, _):
        return Discard

    def NEWLINE(self, _):
        return Discard

    def SPACE(self, _):
        return Discard

    def NOT(self, _):
        return Discard

    def AND(self, _):
        return Discard

    def OR(self, _):
        return Discard

    def IMPLIES(self, _):
        return Discard

    def IFF(self, _):
        return Discard

    def EQUALS(self, _):
        return Discard
