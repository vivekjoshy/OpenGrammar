from opengrammar import _
from typing import List, Union, Optional

from lark import Token


class Operator:
    pass


class Connective(Operator):
    pass


class Quantifier(Operator):
    def __init__(
        self,
        variable: "Variable",
        clause: Union[
            "Predicate", "UnaryConnective", "BinaryConnective", "Quantifier"
        ] = None,
    ):
        self.variable = variable
        self.clause = clause

        # Meta Theoretic
        self.parenthesized = False


class UnaryConnective(Connective):
    def __init__(
        self,
        clause: Union["Predicate", "UnaryConnective", "BinaryConnective", "Quantifier"],
    ):
        self.clause = clause

        # Meta Theoretic
        self.parenthesized = False

        if isinstance(self.clause, Predicate):
            self.atomic = True


class BinaryConnective(Connective):
    def __init__(
        self,
        antecedent: Union[
            "Predicate", "UnaryConnective", "BinaryConnective", "Quantifier"
        ],
        consequent: Union[
            "Predicate", "UnaryConnective", "BinaryConnective", "Quantifier"
        ],
    ):
        self.antecedent = antecedent
        self.consequent = consequent

        # Meta Theoretic
        self.parenthesized = True


class Term:
    pass


class Function(Term):
    def __init__(self, symbol: str, *terms: List["Term"], token: Token):
        self.symbol = symbol
        self.terms = list(terms)
        self.token = token

        # Meta Theoretic
        self.arity = len(self.terms)

    def __repr__(self):
        return f"{self.symbol}({', '.join([repr(t) for t in self.terms])})"

    def __str__(self):
        return _("")

    def __eq__(self, other: "Function"):
        if self.symbol == other.symbol:
            if self.arity == other.arity:
                return True
        return False

    def __hash__(self):
        return hash((self.symbol, self.arity))


class Name(Term):
    def __init__(self, symbol: str):
        self.symbol = symbol

    def __repr__(self):
        return f"{self.symbol}"

    def __str__(self):
        return _("")


class Variable(Term):
    def __init__(self, symbol: str, token: Token):
        self.symbol = symbol
        self.token = token

    def __eq__(self, other: "Variable"):
        return other.symbol == self.symbol

    def __hash__(self):
        return hash(self.symbol)

    def __repr__(self):
        return f"{self.symbol}"

    def __str__(self):
        return _("")


class Predicate:
    def __init__(self, symbol, *terms: List[Term], token: Optional[Token] = None):
        self.symbol = symbol
        self.terms = list(terms)
        self.token = token

        # Meta Theoretic
        self.arity = len(self.terms)

    def __repr__(self):
        return f"{self.symbol}({', '.join([repr(t) for t in self.terms])})"

    def __str__(self):
        return _("")

    def __eq__(self, other: "Predicate"):
        if self.symbol == other.symbol:
            if self.arity == other.arity:
                return True
        return False

    def __hash__(self):
        return hash((self.symbol, self.arity))


class IdentityPredicate(Predicate):
    def __init__(self, *terms: List[Term]):
        self.symbol = "I"
        self.terms = terms
        super().__init__(self.symbol, *self.terms)

    def __repr__(self):
        return " = ".join([repr(t) for t in self.terms])

    def __str__(self):
        return _("")


class ExistentialQuantifier(Quantifier):
    def __repr__(self):
        if self.clause:
            if self.clause.parenthesized:
                return f"∃{repr(self.variable)}{repr(self.clause)}"
            else:
                return f"∃{repr(self.variable)}({repr(self.clause)})"

    def __str__(self):
        return _("")


class UniversalQuantifier(Quantifier):
    def __repr__(self):
        if self.clause:
            if self.clause.parenthesized:
                return f"∀{repr(self.variable)}{repr(self.clause)}"
            else:
                return f"∀{repr(self.variable)}({repr(self.clause)})"

    def __str__(self):
        return _("")


class Negation(UnaryConnective):
    def __repr__(self):
        return f"¬{repr(self.clause)}"

    def __str__(self):
        return _("")


class Conjunction(BinaryConnective):
    def __repr__(self):
        if self.parenthesized:
            return f"({repr(self.antecedent)} ∧ {repr(self.consequent)})"
        else:
            return f"{repr(self.antecedent)} ∧ {repr(self.consequent)}"

    def __str__(self):
        return _("")


class Disjunction(BinaryConnective):
    def __repr__(self):
        if self.parenthesized:
            return f"({repr(self.antecedent)} ∨ {repr(self.consequent)})"
        else:
            return f"{repr(self.antecedent)} ∨ {repr(self.consequent)}"

    def __str__(self):
        return _("")


class Conditional(BinaryConnective):
    def __repr__(self):
        if self.parenthesized:
            return f"({repr(self.antecedent)} → {repr(self.consequent)})"
        else:
            return f"{repr(self.antecedent)} → {repr(self.consequent)}"

    def __str__(self):
        return _("")


class BiConditional(BinaryConnective):
    def __repr__(self):
        if self.parenthesized:
            return f"({repr(self.antecedent)} ↔ {repr(self.consequent)})"
        else:
            return f"{repr(self.antecedent)} ↔ {repr(self.consequent)}"

    def __str__(self):
        return _("")
