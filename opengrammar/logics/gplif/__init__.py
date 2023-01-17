import functools
import os
from collections import defaultdict
from queue import LifoQueue, Queue
from typing import Dict, List, Set, Union

from lark import Lark, UnexpectedInput

from opengrammar.logics.gplif.errors import (
    MissingComma,
    MissingFunction,
    MissingLeftParenthesis,
    MissingRightParenthesis,
    MissingScopedFormula,
    MissingValidOperator,
    MultipleDispatchError,
    UnboundVariableError,
)
from opengrammar.logics.gplif.syntax import (
    BinaryConnective,
    Function,
    Name,
    Predicate,
    Quantifier,
    UnaryConnective,
    Variable,
)
from opengrammar.logics.gplif.transformer import GPLIFTransformer

script_directory = os.path.dirname(__file__)

# Load Formula Grammar
with open(
    os.path.join(script_directory, "grammars/formula.lark"), encoding="utf-8"
) as fp:
    formula_grammar = "".join(fp.readlines())

# Initialize Lark
formula_parser = Lark(grammar=formula_grammar, start="wff", ambiguity="explicit")


class GPLIFFormulaParser:
    def __init__(self, formula: str):
        # Meta
        self.formula = formula

        # Lark Specific
        try:
            self.parse_tree = formula_parser.parse(self.formula)
        except UnexpectedInput as u:
            exc_class = u.match_examples(
                formula_parser.parse,
                {
                    MissingRightParenthesis: [
                        r"P(a",
                        r"P(a, b",
                        r"P(a, b, c",
                        r"∀x(P(a)",
                    ],
                    MissingLeftParenthesis: [r"Pa)", r"Pa, b)"],
                    MissingComma: [r"P(a b)", r"P(a, b c)"],
                    MissingValidOperator: [
                        r"P (a, b)",
                        r"P   (a)",
                        r"P     (a, b, c)",
                        r"∃x∀y(P(a, b)  ∀z(V(z))",
                        r"P(a) ∧ ~Q(b)",
                    ],
                    MissingScopedFormula: [r"∀x∃y()"],
                    MissingFunction: [r"f(a) = P(a)"],
                },
                use_accepts=True,
            )
            raise exc_class(u.get_context(self.formula), u.line, u.column)

        self.transformer = GPLIFTransformer(self.formula)

        # Syntax Tree
        self.syntax_tree = self.transformer.transform(self.parse_tree)

        # Syntax Checks
        self.traverse_syntax_tree()

    def traverse_syntax_tree(self):
        formulas = Queue()
        formulas.put(self.syntax_tree)

        # Syntax Cache
        scope_cache: Dict[Variable, int] = {}
        predicate_cache: List[Predicate] = []
        function_cache: List[Function] = []

        # Top-Down Traversal
        while not formulas.empty():

            # Get Latest Formula
            current_formula = formulas.get()

            # Check Type of Formula
            if isinstance(current_formula, Predicate):

                # Prevent duplicate relations with different arity
                predicate_cache.append(current_formula)
                functions = self.extract_functions(current_formula)
                for f, c in functions.items():
                    function_cache.append(f)

                # Check Unbound Variable
                current_variables = self.variables_from_predicate(current_formula)
                if scope_cache and current_variables:
                    for variable in current_variables:
                        if scope_cache.get(variable):
                            scope_cache[variable] -= 1
                elif current_variables:
                    raise UnboundVariableError(
                        formula=self.formula, variables=current_variables
                    )

                unscoped_variables = []
                for variable in current_variables:
                    if variable not in scope_cache.keys():
                        unscoped_variables.append(variable)

                if unscoped_variables:
                    raise UnboundVariableError(
                        formula=self.formula, variables=unscoped_variables
                    )

            elif isinstance(current_formula, UnaryConnective):
                formulas.put(current_formula.clause)
            elif isinstance(current_formula, BinaryConnective):
                formulas.put(current_formula.antecedent)
                formulas.put(current_formula.consequent)
            elif isinstance(current_formula, Quantifier):
                if current_formula.variable in scope_cache:
                    scope_cache[
                        current_formula.variable
                    ] += self.count_variables_in_scope(current_formula)
                else:
                    scope_cache[
                        current_formula.variable
                    ] = self.count_variables_in_scope(current_formula)
                formulas.put(current_formula.clause)

            # Raise Multiple Dispatch Error for Predicates
            predicates = defaultdict(list)
            for p in predicate_cache:
                predicates[p.symbol].append(p)

            for symbol, predicate_list in predicates.items():
                equal_arity = bool(
                    functools.reduce(lambda a, b: a.arity == b.arity, predicate_list)
                )
                if not equal_arity:
                    raise MultipleDispatchError(
                        formula=self.formula, relations=predicate_list
                    )

            # Raise Multiple Dispatch Error for Functions
            functions = defaultdict(list)
            for f in function_cache:
                functions[f.symbol].append(f)

            for symbol, functions_list in functions.items():
                equal_arity = bool(
                    functools.reduce(lambda a, b: a.arity == b.arity, functions_list)
                )
                if not equal_arity:
                    raise MultipleDispatchError(
                        formula=self.formula, relations=functions_list
                    )

    def count_variables_in_scope(self, quantifier: Quantifier) -> int:
        scope = quantifier.variable
        clause = quantifier.clause
        formulas = Queue()
        formulas.put(clause)

        count = 0

        # Top-Down Traversal
        while not formulas.empty():

            # Get Latest Formula
            current_formula = formulas.get()

            if isinstance(current_formula, Predicate):
                variables = list(
                    filter(
                        lambda _: _ == scope,
                        self.variables_from_predicate(current_formula),
                    )
                )
                count += len(variables)
            elif isinstance(current_formula, UnaryConnective):
                formulas.put(current_formula.clause)
            elif isinstance(current_formula, BinaryConnective):
                formulas.put(current_formula.antecedent)
                formulas.put(current_formula.consequent)
            elif isinstance(current_formula, Quantifier):
                formulas.put(current_formula.clause)

        return count

    def variables_from_predicate(self, predicate: Predicate):
        variables = []
        variables.extend(
            list(filter(lambda _: isinstance(_, Variable), predicate.terms))
        )
        outer_functions = set(
            filter(lambda _: isinstance(_, Function), predicate.terms)
        )
        for outer_function in outer_functions:
            if all(isinstance(_, Name) for _ in outer_function.terms):
                break

            if any(isinstance(_, Variable) for _ in outer_function.terms):
                variables.extend(
                    list(
                        filter(lambda _: isinstance(_, Variable), outer_function.terms)
                    )
                )

            if all(not isinstance(_, Function) for _ in outer_function.terms):
                break

            if any(isinstance(_, Function) for _ in outer_function.terms):
                inner_functions = set(
                    filter(lambda _: isinstance(_, Function), outer_function.terms)
                )
                for inner_function in inner_functions:
                    inner_variables = self.variables_from_function(inner_function)
                    variables.extend(inner_variables)

        ordered_variables = list(dict.fromkeys(variables))
        return ordered_variables

    def variables_from_function(self, function: Function):
        variables = []
        variables.extend(
            list(filter(lambda _: isinstance(_, Variable), function.terms))
        )
        outer_functions = set(filter(lambda _: isinstance(_, Function), function.terms))
        if not outer_functions:
            ordered_variables = list(dict.fromkeys(variables))
            return ordered_variables

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

                    if isinstance(t, Variable):
                        variables.append(t)

        ordered_variables = list(dict.fromkeys(variables))
        return ordered_variables

    def extract_functions(self, relation: Union[Predicate, Function]):
        outer_functions: Set[Function] = set(
            filter(lambda _: isinstance(_, Function), relation.terms)
        )
        function_count: Dict[Function, int] = dict()
        while len(outer_functions):
            for outer_function in outer_functions:
                if any(isinstance(_, Function) for _ in outer_function.terms):

                    outer_functions = set(
                        filter(
                            lambda _: isinstance(_, Function),
                            outer_function.terms,
                        )
                    )

                    function_count[outer_function] = 1 + function_count.get(
                        outer_function, 0
                    )
                else:
                    function_count[outer_function] = 1 + function_count.get(
                        outer_function, 0
                    )
                    outer_functions = set()
        return function_count
