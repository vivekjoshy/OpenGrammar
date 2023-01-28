from typing import List, Union

from lark import UnexpectedInput

from opengrammar import _
from opengrammar.logics.gplif.syntax import Function, Predicate, Variable


class UnboundVariableError(UnexpectedInput):
    def __init__(self, formula: str, variables: List[Variable]):
        self.formula = formula
        self.variables = variables
        self.ordered_variables = self.variables
        self.column_start = (
            min(self.ordered_variables, key=lambda _: int(_.token.column)).token.column
            - 1
        )
        self.formula_length = len(self.formula)
        self.error_positions = str(" " * len(formula))
        self.indicator = "^"

        for variable in self.variables:
            self.error_positions = (
                self.error_positions[: variable.token.column - 1]
                + self.indicator
                + self.error_positions[variable.token.column :]
            )

        self.message = _(
            "Variables {variables} starting at column {column_start} "
            "are not bound by any scope.\n\n"
            "{formula}\n"
            "{error_positions}"
        ).format(
            variables=self.variables,
            column_start=self.column_start + 1,
            formula=self.formula,
            error_positions=self.error_positions,
        )
        super().__init__(self.message)


class MultipleDispatchError(UnexpectedInput):
    def __init__(self, formula: str, relations: Union[List[Predicate], List[Function]]):
        self.formula = formula
        self.relations = relations
        self.formula_length = len(self.formula)
        self.column_start = (
            min(self.relations, key=lambda _: int(_.token.column)).token.column - 1
        )
        self.formula_length = len(self.formula)
        self.error_positions = " " * self.formula_length
        self.indicator = "^"

        for relation in self.relations:
            self.error_positions = (
                self.error_positions[: relation.token.column - 1]
                + self.indicator
                + self.error_positions[relation.token.column :]
            )

        if isinstance(self.relations[0], Function):
            self.message = _(
                "Function {function} starting at column {column_start} "
                "is defined multiple times with mismatched arity.\n\n"
                "{formula}\n"
                "{error_positions}"
            ).format(
                function=self.relations[0].symbol,
                column_start=self.column_start + 1,
                formula=self.formula,
                error_positions=self.error_positions,
            )
        else:
            self.message = _(
                "Predicate {predicate} at column {column_start} "
                "is defined multiple times with mismatched arity.\n\n"
                "{formula}\n"
                "{error_positions}"
            ).format(
                predicate=self.relations[0].symbol,
                column_start=self.column_start + 1,
                formula=self.formula,
                error_positions=self.error_positions,
            )

        super().__init__(self.message)


class GPLIFSyntaxError(Exception):
    def __str__(self):
        context, line, column = self.args
        return f"{self.label} at column {column}.\n\n{context}"


class MissingComma(GPLIFSyntaxError):
    label = "Missing Comma"


class MissingRightParenthesis(GPLIFSyntaxError):
    label = "Missing Right Parenthesis"


class MissingLeftParenthesis(GPLIFSyntaxError):
    label = "Missing Left Parenthesis"


class MissingValidOperator(GPLIFSyntaxError):
    label = "Missing Valid Operator"


class MissingScopedFormula(GPLIFSyntaxError):
    label = "Missing Formula for Quantifier"


class MissingFunction(GPLIFSyntaxError):
    label = "Missing Function"


class MissingFormula(GPLIFSyntaxError):
    label = "Missing Formula"


class TranslationError(Exception):
    pass
