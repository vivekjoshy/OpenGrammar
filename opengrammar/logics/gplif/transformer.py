from collections import Counter
from functools import reduce
from typing import List, Optional, Type, Union

from lark import Transformer

from opengrammar.logics.gplif.errors import MultipleDispatchError, UnboundVariableError
from opengrammar.logics.gplif.syntax import (
    BiConditional,
    BinaryConnective,
    Conditional,
    Conjunction,
    Connective,
    Disjunction,
    ExistentialQuantifier,
    Function,
    IdentityPredicate,
    Name,
    Negation,
    Operator,
    Predicate,
    Quantifier,
    Term,
    UnaryConnective,
    UniversalQuantifier,
    Variable,
)


class GPLIFTransformer(Transformer):
    def __init__(self, formula):
        self.formula = formula

        super().__init__()

    def NAME(self, items):
        symbol = str(items[0])
        return Name(symbol=symbol)

    def VARIABLE(self, items):
        symbol = str(items[0])
        return Variable(symbol=symbol, token=items)

    def function(self, items):
        symbol = str(items[0].value)
        terms = list(filter(lambda _: isinstance(_, Term), items))
        function = Function(symbol, *terms, token=items[0])
        return function

    def term(self, items):
        if isinstance(items[0], Term):
            return items[0]

    def predicate(self, items):
        symbol = str(items[0].value)
        terms = list(filter(lambda _: isinstance(_, Term), items))
        predicate = Predicate(symbol, *terms, token=items[0])
        return predicate

    def atomic_wff(self, items):
        if len(items) == 1:
            clause = items[0]
            return clause

    def compound_wff(self, items):
        clause = items[0]
        return clause

    def quantified_wff(self, items):
        quantifiers = list(
            reversed(list(filter(lambda _: isinstance(_, Quantifier), items)))
        )
        deepest_scope = list(
            filter(lambda _: isinstance(_, (Predicate, Connective)), items)
        )[0]
        if not quantifiers:
            return deepest_scope

        current_scope = None
        for quantifier in quantifiers:
            if isinstance(quantifier, Quantifier):
                if current_scope:
                    outer_scope = quantifier
                    outer_scope.clause = current_scope
                    current_scope = outer_scope
                else:
                    current_scope = quantifier
                    current_scope.clause = deepest_scope
        return current_scope

    def quantified_scope(self, items):
        return items[0]

    def curly_quantifiers(self, items):
        if items[0].type == "UNIVERSAL_QUANTIFIER_SYMBOL":
            variable = list(filter(lambda _: isinstance(_, Variable), items))[0]
            return UniversalQuantifier(variable=variable)
        elif items[0].type == "EXISTENTIAL_QUANTIFIER_SYMBOL":
            variable = list(filter(lambda _: isinstance(_, Variable), items))[0]
            return ExistentialQuantifier(variable=variable)

    def quantifiers(self, items):
        if items[0].type == "UNIVERSAL_QUANTIFIER_SYMBOL":
            variable = list(filter(lambda _: isinstance(_, Variable), items))[0]
            return UniversalQuantifier(variable=variable)
        elif items[0].type == "EXISTENTIAL_QUANTIFIER_SYMBOL":
            variable = list(filter(lambda _: isinstance(_, Variable), items))[0]
            return ExistentialQuantifier(variable=variable)

    def identity_wff(self, items):
        terms = list(filter(lambda _: isinstance(_, Term), items))
        if terms:
            predicate = IdentityPredicate(*terms)
            return predicate
        else:
            clause = items[0]
            return clause

    def negated_wff(self, items):
        if len(items) == 2:
            negated_statement = Negation(clause=items[1])
            return negated_statement
        elif len(items) > 2:
            negated_statement = Negation(clause=items[-1])
            return negated_statement
        else:
            return items[0]

    def conjunctive_wff(self, items):
        if len(items) > 1:
            statements = list(
                reversed(
                    list(filter(lambda _: isinstance(_, (Operator, Predicate)), items))
                )
            )
            return reduce(
                lambda a, c: scope_binary_connective(c, a, Conjunction), statements
            )
        else:
            return items[0]

    def disjunctive_wff(self, items):
        if len(items) > 1:
            statements = list(
                reversed(
                    list(filter(lambda _: isinstance(_, (Operator, Predicate)), items))
                )
            )
            return reduce(
                lambda a, c: scope_binary_connective(c, a, Disjunction), statements
            )
        else:
            return items[0]

    def conditional_wff(self, items):
        if len(items) > 1:
            statements = list(
                reversed(
                    list(filter(lambda _: isinstance(_, (Operator, Predicate)), items))
                )
            )
            return reduce(
                lambda a, c: scope_binary_connective(c, a, Conditional), statements
            )
        else:
            return items[0]

    def biconditional_wff(self, items):
        if len(items) > 1:
            statements = list(
                reversed(
                    list(filter(lambda _: isinstance(_, (Operator, Predicate)), items))
                )
            )
            return reduce(
                lambda a, c: scope_binary_connective(c, a, BiConditional), statements
            )
        else:
            return items[0]


def scope_binary_connective(
    a: Union[Operator, Predicate],
    b: Union[Operator, Predicate],
    m: Type[BinaryConnective],
):
    connective = m(a, b)
    return connective
