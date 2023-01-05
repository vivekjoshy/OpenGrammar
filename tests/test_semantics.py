import pytest

from opengrammar.logics.gplif import (
    GPLIFFormulaParser,
    GPLIFModel,
    MultipleDispatchError,
    UnboundVariableError,
)


def test_unbound_variables():

    # Unbound Variables Errors
    input_formulas = [
        r"∀x(P(a) ∧ ∀y(Q(f(b, g(h(y), j(k(z)))))))",
        r"¬P(y)",
        r"¬P(a) ∧ Q(f(y))",
        r"∀x(P(y) ∧ ∀y(Q(f(b, g(h(c), j(d))))))",
        r"\exists{x} \forall{y} (P(f(x, y)) \vee \forall{z} ((V(u) \iff \neg R(a)) \wedge B(a)))",
    ]

    for formula in input_formulas:
        parser = GPLIFFormulaParser(formula=formula)
        with pytest.raises(UnboundVariableError):
            model = GPLIFModel(parsers=[parser])


def test_multiple_dispatch_error():
    # Multiple Dispatch Error
    input_formulas = [
        r"P(a) ∧ P(a, b)",
        r"¬P(f(a, b), c) ∧ P(a, f(b))",
        r"\exists{x} \forall{y} (P(f(x, y)) \vee \forall{z} ((V(f(z)) \iff \neg R(a)) \wedge B(a)))",
    ]
    for formula in input_formulas:
        parser = GPLIFFormulaParser(formula=formula)
        with pytest.raises(MultipleDispatchError):
            model = GPLIFModel(parsers=[parser])
