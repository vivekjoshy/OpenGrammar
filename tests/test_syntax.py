import pytest
from lark import Token

from opengrammar.logics.gplif import (
    Function,
    GPLIFFormulaParser,
    MultipleDispatchError,
    UnboundVariableError,
)
from opengrammar.logics.gplif.syntax import Name, Predicate


def test_wff():

    # Well Formed Formulas
    input_formulas = [
        r"f(a, h(k(l(b)))) = g(c, j(d))",
        r"(P(a) ∨ Q(b)) ∧ (R(c) → S(d))",
        r"(P(a) ∨ f(b, c) = g(d)) ∧ (R(e) → S(f))",
        r"∀x∃y(¬P(f(g(a, x))) ∨ Q(y) ∧ R(x))",
        r"P(a) ∨ Q(b) ∨ R(c)",
        r"\exists{x} \forall{y} (P(f(x, y)) \vee \forall{z} ((V(z) \iff \neg R(a)) \wedge B(a)))",
        r"\forall{x} (R(x) \implies H(x)) \land \exists{x} (G(x) \land \lnot H(x))",
        r"∀x∃y((P(x) ∧ Q(y)) ∧ (R(y)))",
        r"∀x(P(x) ∧ Q(a)) ∧ ∃y(P(y) ∧ Q(a))",
        r"∃x∀y(P(f(a, b)) ∨ ∀z(V(z) ↔ ¬R(a)) ∧ B(a))",
        r"∃x∀y(P(f(x, y), a, b, c) → ∀z(V(z) → ¬R(a)) ∧ B(a))",
        r"P(a) ∨ ¬(Q(b) ∨ R(c))",
        r"P(a) ∨ ¬∀y(Q(b) ∨ R(c))",
    ]
    output_formulas = [
        r"f(a, h(k(l(b)))) = g(c, j(d))",
        r"((P(a) ∨ Q(b)) ∧ (R(c) → S(d)))",
        r"((P(a) ∨ f(b, c) = g(d)) ∧ (R(e) → S(f)))",
        r"∀x(∃y(¬P(f(g(a, x))) ∨ (Q(y) ∧ R(x))))",
        r"(P(a) ∨ (Q(b) ∨ R(c)))",
        r"∃x(∀y(P(f(x, y)) ∨ ∀z((V(z) ↔ ¬R(a)) ∧ B(a))))",
        r"(∀x(R(x) → H(x)) ∧ ∃x(G(x) ∧ ¬H(x)))",
        r"∀x(∃y((P(x) ∧ Q(y)) ∧ R(y)))",
        r"(∀x(P(x) ∧ Q(a)) ∧ ∃y(P(y) ∧ Q(a)))",
        r"∃x(∀y(P(f(a, b)) ∨ (∀z(V(z) ↔ ¬R(a)) ∧ B(a))))",
        r"∃x(∀y(P(f(x, y), a, b, c) → (∀z(V(z) → ¬R(a)) ∧ B(a))))",
        r"(P(a) ∨ ¬(Q(b) ∨ R(c)))",
        r"(P(a) ∨ ¬∀y(Q(b) ∨ R(c)))",
    ]

    assert len(input_formulas) == len(output_formulas)

    for input_formula, output_formula in zip(input_formulas, output_formulas):
        parser = GPLIFFormulaParser(formula=input_formula)
        output_syntax_tree = parser.syntax_tree
        assert repr(output_syntax_tree) == output_formula


def test_equality():
    # Manual Tests for Equality
    terms = [Name("a"), Name("b"), Name("c")]

    f = Function("f", *terms, token=Token("FUNCTION_NAME", "f"))
    g = Function("g", *terms, token=Token("FUNCTION_NAME", "g"))
    h = Function("h", *terms[:1], token=Token("FUNCTION_NAME", "h"))

    assert f != g
    assert f == f
    assert f != h

    p = Predicate("P", *terms, token=Token("PREDICATE_NAME", "P"))
    q = Predicate("Q", *terms, token=Token("PREDICATE_NAME", "Q"))
    r = Predicate("R", *terms[:1], token=Token("PREDICATE_NAME", "R"))

    assert p != q
    assert p == p
    assert p != r


def test_unbound_variables():

    # Unbound Variables Errors
    input_formulas = [
        r"∀x(P(a) ∧ ∀y(Q(f(b, g(h(y), j(k(z)))))))",
        r"¬P(y)",
        r"¬P(a) ∧ Q(f(y))",
        r"∀x(P(y) ∧ ∀y(Q(f(b, g(h(c), j(d))))))",
        r"\exists{x} \forall{y} (P(f(x, y)) \vee \forall{z} ((V(u) \iff \neg R(a)) \wedge B(a)))",
        r"∀x(P(f(g(h(j(y))))))",
    ]

    for formula in input_formulas:
        with pytest.raises(UnboundVariableError):
            parser = GPLIFFormulaParser(formula=formula)


def test_multiple_dispatch_error():
    # Multiple Dispatch Error
    input_formulas = [
        r"P(a) ∧ P(a, b)",
        r"¬P(f(a, b), c) ∧ P(a, f(b))",
        r"\exists{x} \forall{y} (P(f(x, y)) \vee \forall{z} ((V(f(z)) \iff \neg R(a)) \wedge B(a)))",
    ]
    for formula in input_formulas:
        with pytest.raises(MultipleDispatchError):
            parser = GPLIFFormulaParser(formula=formula)
