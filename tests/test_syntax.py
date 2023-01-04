from opengrammar.logics.gplif import GPLIFFormulaParser


def test_syntax():

    # Well Formed Formulas
    input_formulas = [
        r"P(a) ∨ Q(b) ∨ R(c)",
        r"\exists{x} \forall{y} (P(f(x, y)) \vee \forall{z} ((V(z) \iff \neg R(a)) \wedge B(a)))",
        r"\forall{x} (R(x) \implies H(x)) \land \exists{x} (G(x) \land \lnot H(x))",
        r"∀x∃y((P(x) ∧ Q(y)) ∧ (R(y)))",
        r"∀x∃y(¬P(f(g(a, x))) ∨ Q(y) ∧ R(x))",
        r"∀x(P(x) ∧ Q(a)) ∧ ∃y(P(y) ∧ Q(a))",
        r"∃x∀y(P(f(a, b)) ∨ ∀z(V(z) ↔ ¬R(a)) ∧ B(a))",
        r"∃x∀y(P(f(x, y), a, b, c) → ∀z(V(z) → ¬R(a)) ∧ B(a))",
        r"(P(a) ∨ Q(b)) ∧ (R(c) → S(d))",
        r"(P(a) ∨ f(b, c) = g(d)) ∧ (R(e) → S(f))",
        r"f(a, h(k(l(b)))) = g(c, j(d))",
        r"P(a) ∨ ¬(Q(b) ∨ R(c))",
        r"P(a) ∨ ¬∀y(Q(b) ∨ R(c))",
    ]
    output_formulas = [
        r"(P(a) ∨ (Q(b) ∨ R(c)))",
        r"∃x(∀y(P(f(x, y)) ∨ ∀z((V(z) ↔ ¬R(a)) ∧ B(a))))",
        r"(∀x(R(x) → H(x)) ∧ ∃x(G(x) ∧ ¬H(x)))",
        r"∀x(∃y((P(x) ∧ Q(y)) ∧ R(y)))",
        r"∀x(∃y(¬P(f(g(a, x))) ∨ (Q(y) ∧ R(x))))",
        r"(∀x(P(x) ∧ Q(a)) ∧ ∃y(P(y) ∧ Q(a)))",
        r"∃x(∀y(P(f(a, b)) ∨ (∀z(V(z) ↔ ¬R(a)) ∧ B(a))))",
        r"∃x(∀y(P(f(x, y), a, b, c) → (∀z(V(z) → ¬R(a)) ∧ B(a))))",
        r"((P(a) ∨ Q(b)) ∧ (R(c) → S(d)))",
        r"((P(a) ∨ f(b, c) = g(d)) ∧ (R(e) → S(f)))",
        r"f(a, h(k(l(b)))) = g(c, j(d))",
        r"(P(a) ∨ ¬(Q(b) ∨ R(c)))",
        r"(P(a) ∨ ¬∀y(Q(b) ∨ R(c)))",
    ]

    for input_formula, output_formula in zip(input_formulas, output_formulas):
        parser = GPLIFFormulaParser(formula=input_formula)
        output_syntax_tree = parser.syntax_tree
        assert repr(output_syntax_tree) == output_formula
