import functools
import os
from queue import Queue
from typing import Dict, List, Set, Union

from lark import Lark

from opengrammar.logics.gplif.errors import MultipleDispatchError, UnboundVariableError
from opengrammar.logics.gplif.syntax import (
    BinaryConnective,
    Function,
    Predicate,
    Quantifier,
    UnaryConnective,
    Variable,
)
from opengrammar.logics.gplif.transformer import GPLIFTransformer

script_directory = os.path.dirname(__file__)
with open(
    os.path.join(script_directory, "grammars/formula.lark"), encoding="utf-8"
) as fp:
    formula_grammar = "".join(fp.readlines())

parser = Lark(grammar=formula_grammar, start="wff", ambiguity="explicit")


class GPLIFFormulaParser:
    def __init__(self, formula: str):
        # Meta
        self.formula = formula

        # Lark Specific
        self.parse_tree = parser.parse(self.formula)
        self.transformer = GPLIFTransformer(self.formula)

        # Syntax Tree
        self.syntax_tree = self.transformer.transform(self.parse_tree)


class GPLIFModel:
    def __init__(self, parsers: List[GPLIFFormulaParser]):
        self.parsers = parsers

        for parser in self.parsers:
            self.current_wff = parser.formula
            self.traverse_syntax_tree(parser)

    def traverse_syntax_tree(self, parser: GPLIFFormulaParser):
        formulas = Queue()
        formulas.put(parser.syntax_tree)

        # Semantic Cache
        scope: List[Variable] = []
        predicates: Dict[Predicate, int] = dict()
        functions: Dict[Function, int] = dict()

        # Top-Down Traversal
        while not formulas.empty():

            # Get Latest Formula
            current_formula = formulas.get()

            # Check Type of Formula
            if isinstance(current_formula, Predicate):

                # Populate Predicate Arity Cache
                predicates[current_formula] = 1 + predicates.get(current_formula, 0)
                predicate_symbol = current_formula.symbol
                current_predicates = list(
                    filter(lambda _: _.symbol == predicate_symbol, predicates.keys())
                )
                equal_arity = (
                    bool(
                        functools.reduce(
                            lambda a, b: a.arity == b.arity, current_predicates
                        )
                    )
                    if current_predicates
                    else True
                )
                for current_predicate in current_predicates:
                    if predicates.get(current_predicate) and not equal_arity:
                        raise MultipleDispatchError(
                            formula=self.current_wff, symbol=current_formula
                        )

                # Populate Function Arity Cache
                for key, val in self.extract_functions(current_formula).items():
                    functions[key] = val

                for fn in functions:
                    functions[fn] = 1 + functions.get(fn, 0)
                    fn_symbol = fn.symbol
                    current_functions = list(
                        filter(lambda _: _.symbol == fn_symbol, functions.keys())
                    )
                    equal_arity = (
                        bool(
                            functools.reduce(
                                lambda a, b: a.arity == b.arity, current_functions
                            )
                        )
                        if current_functions
                        else True
                    )
                    for current_function in current_functions:
                        if functions.get(current_function) and not equal_arity:
                            raise MultipleDispatchError(
                                formula=self.current_wff, symbol=fn
                            )

                # Check Unbound Variable
                current_variables = self.variables_from_predicate(current_formula)
                scope = list(dict.fromkeys(scope))
                if scope:
                    if current_variables:
                        for variable in current_variables:
                            if variable in scope:
                                scope.remove(variable)
                            else:
                                raise UnboundVariableError(
                                    formula=self.current_wff, variables=[variable]
                                )
                    else:
                        pass
                else:
                    if current_variables:
                        raise UnboundVariableError(
                            formula=self.current_wff, variables=current_variables
                        )
            elif isinstance(current_formula, UnaryConnective):
                formulas.put(current_formula.clause)
            elif isinstance(current_formula, BinaryConnective):
                formulas.put(current_formula.antecedent)
                formulas.put(current_formula.consequent)
            elif isinstance(current_formula, Quantifier):
                scope.append(current_formula.variable)
                formulas.put(current_formula.clause)

    def variables_from_predicate(self, predicate: Predicate):
        variables = []
        variables.extend(
            list(filter(lambda _: isinstance(_, Variable), predicate.terms))
        )
        outer_functions = set(
            filter(lambda _: isinstance(_, Function), predicate.terms)
        )
        for outer_function in outer_functions:
            if any(isinstance(_, Variable) for _ in outer_function.terms):
                variables.extend(
                    list(
                        filter(lambda _: isinstance(_, Variable), outer_function.terms)
                    )
                )

            if any(isinstance(_, Function) for _ in outer_function.terms):
                inner_functions = set(
                    filter(lambda _: isinstance(_, Function), outer_function.terms)
                )
                for inner_function in inner_functions:
                    variables.extend(self.variables_from_function(inner_function))

        ordered_variables = list(dict.fromkeys(variables))
        return ordered_variables

    def variables_from_function(self, function: Function):
        variables = []
        variables.extend(
            list(filter(lambda _: isinstance(_, Variable), function.terms))
        )
        outer_functions = set(filter(lambda _: isinstance(_, Function), function.terms))
        while len(outer_functions):
            for outer_function in outer_functions:
                if any(isinstance(_, Variable) for _ in outer_function.terms):
                    variables.extend(
                        list(
                            filter(
                                lambda _: isinstance(_, Variable), outer_function.terms
                            )
                        )
                    )
                    outer_functions = outer_functions.difference({outer_function})

                if any(isinstance(_, Function) for _ in outer_function.terms):
                    outer_functions = set(
                        filter(lambda _: isinstance(_, Function), outer_function.terms)
                    )
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
