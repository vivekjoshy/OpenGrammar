from queue import LifoQueue
from typing import List, Optional, Set, Union

from lark import Token

from opengrammar import _


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
    def __init__(
        self,
        symbol: str,
        *terms: List[Union["Name", "Variable", "Function"]],
        token: Token,
    ):
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

    def _extract_items(
        self, type: Union["Name.__class__", "Variable.__class__", "Function.__class__"]
    ):
        function_items = dict.fromkeys(
            filter(lambda _: isinstance(_, type), self.terms)
        )
        outer_functions = set(filter(lambda _: isinstance(_, Function), self.terms))
        if not outer_functions:
            ordered_names = list(dict.fromkeys(function_items))
            return ordered_names

        function_stack = LifoQueue()
        for f in outer_functions:
            function_stack.put(f)

        while not function_stack.empty():

            inner_term = function_stack.get()

            # Update Functions
            if isinstance(inner_term, Function):
                for t in inner_term.terms:
                    if isinstance(t, Function):
                        function_stack.put(t)
                    elif isinstance(t, type):
                        function_items[t] = None
        ordered_names = set(function_items.keys())
        return ordered_names

    @property
    def names(self):
        return self._extract_items(type=Name)

    @property
    def variables(self):
        return self._extract_items(type=Variable)

    @property
    def functions(self):
        return self._extract_items(type=Function)


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
    def __init__(
        self,
        symbol,
        *terms: List[Union[Name, Variable, Function]],
        token: Optional[Token] = None,
    ):
        self.symbol = symbol
        self.terms = list(terms)
        self.token = token

        # Meta Theoretic
        self.arity = len(self.terms)
        self.parenthesized = False

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
        return hash(self.symbol)

    def _extract_items(
        self, type: Union[Name.__class__, Variable.__class__, Function.__class__]
    ):
        predicate_items = dict.fromkeys(
            filter(lambda _: isinstance(_, type), self.terms)
        )
        outer_functions: Set[Function] = set(
            filter(lambda _: isinstance(_, Function), self.terms)
        )
        for outer_function in outer_functions:
            if type == Name:
                predicate_items.update(dict.fromkeys(outer_function.names))
            elif type == Variable:
                predicate_items.update(dict.fromkeys(outer_function.variables))
            elif type == Function:
                predicate_items.update(dict.fromkeys(outer_function.functions))
        ordered_items = set(predicate_items.keys())
        return ordered_items

    @property
    def names(self):
        return self._extract_items(type=Name)

    @property
    def variables(self):
        return self._extract_items(type=Variable)

    @property
    def functions(self):
        return self._extract_items(type=Function)


class IdentityPredicate(Predicate):
    def __init__(self, *terms: List[Term]):
        self.symbol = "I"
        self.terms = terms
        self.arity = len(terms)
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
        return f"({repr(self.antecedent)} ∧ {repr(self.consequent)})"

    def __str__(self):
        return _("")


class Disjunction(BinaryConnective):
    def __repr__(self):
        return f"({repr(self.antecedent)} ∨ {repr(self.consequent)})"

    def __str__(self):
        return _("")


class Conditional(BinaryConnective):
    def __repr__(self):
        return f"({repr(self.antecedent)} → {repr(self.consequent)})"

    def __str__(self):
        return _("")


class BiConditional(BinaryConnective):
    def __repr__(self):
        return f"({repr(self.antecedent)} ↔ {repr(self.consequent)})"

    def __str__(self):
        return _("")
