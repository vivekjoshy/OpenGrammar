from pathlib import Path

from opengrammar.parser import MetaSyntaxParser


def test_ast() -> None:
    """
    Tests the Meta Syntax AST
    """

    sample_ug = Path(__file__).parent / "sample.ug"
    with open(sample_ug, "r") as f:
        sample_grammar = f.read()

    meta_syntax_parser: MetaSyntaxParser = MetaSyntaxParser()
    meta_ast = meta_syntax_parser.parse(text=sample_grammar)

    for rule_index, rule in enumerate(meta_ast.rules):
        for lhs_rule_index, lhs_rule in enumerate(rule.lhs.rules):
            assert lhs_rule.number == lhs_rule_index + 1
