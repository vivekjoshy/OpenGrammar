from typing import Dict, List, Set, Union

from lark import UnexpectedInput

from opengrammar import _
from opengrammar.logics.gplif.syntax import Function, Predicate, Variable


class UnboundVariableError(UnexpectedInput):
    def __init__(self, formula: str, variables: List[Variable]):
        self.formula = formula
        self.variables = variables
        self.ordered_variables = self.variables
        self.formula_length = len(self.formula)
        self.column_start = self.ordered_variables[0].token.column - 1
        self.column_end = self.ordered_variables[-1].token.end_column - 1
        self.error_length = abs(self.column_start - self.column_end)
        self.spaces = " " * self.column_start
        self.indicator = "^" * self.error_length

        self.message = _(
            "Variables {variables} at column {column_start} "
            "are not bound by any scope.\n\n"
            "{formula}\n"
            "{spaces}{indicator}"
        ).format(
            variables=self.variables,
            column_start=self.column_start + 1,
            formula=self.formula,
            spaces=self.spaces,
            indicator=self.indicator,
        )
        super().__init__(self.message)


class MultipleDispatchError(UnexpectedInput):
    def __init__(self, formula: str, symbol: Union[Function, Predicate]):
        self.formula = formula
        self.symbol = symbol
        self.formula_length = len(self.formula)
        self.column_start = self.symbol.token.column - 1
        self.column_end = self.symbol.token.end_column - 1
        self.error_length = abs(self.column_start - self.column_end)
        self.spaces = " " * self.column_start
        self.indicator = "^" * self.error_length

        if isinstance(self.symbol, Function):
            self.message = _(
                "Function {function} at column {column_start} "
                "is defined multiple times with mismatched arity.\n\n"
                "{formula}\n"
                "{spaces}{indicator}"
            ).format(
                function=self.symbol.symbol,
                column_start=self.column_start + 1,
                formula=self.formula,
                spaces=self.spaces,
                indicator=self.indicator,
            )
        else:
            self.message = _(
                "Predicate {predicate} at column {column_start} "
                "is defined multiple times with mismatched arity.\n\n"
                "{formula}\n"
                "{spaces}{indicator}"
            ).format(
                predicate=self.symbol.symbol,
                column_start=self.column_start + 1,
                formula=self.formula,
                spaces=self.spaces,
                indicator=self.indicator,
            )

        super().__init__(self.message)
