"""
The meta-syntax parser.
"""
from enum import Enum
from typing import List, Optional, Union

from rich.tree import Tree


class PostfixType(Enum):
    """
    The kind of postfix operators available.
    """

    ZERO_OR_MORE = "*"
    ONE_OR_MORE = "+"
    OPTIONAL = "?"


class NonTerminal:
    """
    A non-terminal symbol.
    """

    def __init__(self, symbol: str, postfix: Optional[PostfixType] = None) -> None:
        """
        Initializes the NonTerminal.

        :param symbol: The symbol of the non-terminal.
        :param postfix: The postfix operator.
        """
        self.symbol: str = symbol
        self.postfix: Optional[PostfixType] = postfix

    def __rich__(self) -> str:
        if self.postfix:
            string = (
                f"[bright_green]NonTerminal[/bright_green] "
                f"[[yellow]{self.postfix.value}[/yellow]]: "
                f"[bright_blue]{self.symbol}[/bright_blue]"
            )
        else:
            string = (
                f"[bright_green]NonTerminal[/bright_green]: "
                f"[bright_blue]{self.symbol}[/bright_blue]"
            )
        return string

    def __repr__(self) -> str:
        return f"NonTerminal(symbol={self.symbol}, postfix={self.postfix})"


class Terminal:
    """
    A terminal symbol.
    """

    def __init__(self, symbol: str) -> None:
        """
        Initializes the Terminal.

        :param symbol: The symbol of the terminal.
        """
        self.symbol: str = symbol

    def __rich__(self) -> str:
        string = (
            f"[bright_red]Terminal[/bright_red]   : "
            f"[bright_blue]{self.symbol}[/bright_blue]"
        )
        return string

    def __repr__(self) -> str:
        return f"Terminal(symbol={self.symbol})"


class Disjunction:
    """
    A disjunction.
    """

    def __init__(
        self,
        antecedent: "Operands",
        consequent: "Operands",
        postfix: Optional[PostfixType] = None,
    ) -> None:
        """
        The disjunction of two operands.

        :param antecedent: An antecedent.
        :param consequent: A consequent.
        :param postfix: The postfix operator.
        """
        self.antecedent = antecedent
        self.consequent = consequent
        self.postfix = postfix

    def __rich__(self) -> Tree:
        if self.postfix:
            string = f"Or [[yellow]{self.postfix.value}[/yellow]]"
        else:
            string = "Or"

        tree = Tree(string)
        tree.add(self.antecedent)
        tree.add(self.consequent)
        return tree

    def __repr__(self) -> str:
        return (
            f"Disjunction("
            f"antecedent={self.antecedent}, "
            f"consequent={self.consequent}, "
            f"postfix={self.postfix})"
        )


class Conjunction:
    """
    A conjunction.
    """

    def __init__(
        self,
        antecedent: "Operands",
        consequent: "Operands",
        postfix: Optional[PostfixType] = None,
    ) -> None:
        """
        The conjunction of two operands.

        :param antecedent: An antecedent.
        :param consequent: A consequent.
        :param postfix: A postfix operator.
        """
        self.antecedent = antecedent
        self.consequent = consequent
        self.postfix = postfix

    def __rich__(self) -> Tree:
        if self.postfix:
            string = f"And [[yellow]{self.postfix.value}[/yellow]]"
        else:
            string = "And"

        tree = Tree(string)
        tree.add(self.antecedent)
        tree.add(self.consequent)
        return tree

    def __repr__(self) -> str:
        return (
            f"Conjunction("
            f"antecedent={self.antecedent}, "
            f"consequent={self.consequent}, "
            f"postfix={self.postfix})"
        )


class LHS:
    """
    The left-hand side of a rule.
    """

    def __init__(self, rule: Union[NonTerminal, Conjunction, Disjunction]):
        """
        Initializes the LHS.

        :param rule: A rule.
        """
        self.rule = rule

    def __rich__(self) -> Tree:
        tree = Tree("LHS")
        tree.add(self.rule)
        return tree

    def __repr__(self) -> str:
        return f"LHS(rule={self.rule})"


class RHS:
    """
    The right-hand side of a rule.
    """

    def __init__(self, rule: "Operands"):
        """
        Initializes the RHS.

        :param rule: A rule.
        """
        self.rule = rule

    def __rich__(self) -> Tree:
        tree = Tree("RHS")
        tree.add(self.rule)
        return tree

    def __repr__(self) -> str:
        return f"RHS(rule={self.rule})"


class Rule:
    """
    A rule.
    """

    def __init__(self, lhs: LHS, rhs: RHS, number: Optional[int] = None) -> None:
        """
        Initializes the Rule.

        :param lhs: The left-hand side of the rule.
        :param rhs: The right-hand side of the rule.
        :param number: The rule number.
        """
        self.lhs: LHS = lhs
        self.rhs: RHS = rhs
        self.number: Optional[int] = number

    def __rich__(self) -> Tree:
        if self.number:
            string = (
                f"[bright_magenta]Rule[/bright_magenta] "
                f"[[yellow]{self.number}[/yellow]]"
            )
        else:
            string = f"[bright_magenta]Rule[/bright_magenta]"
        tree = Tree(string)
        tree.add(self.lhs)
        tree.add(self.rhs)
        return tree

    def __repr__(self) -> str:
        return f"Rule(lhs={self.lhs}, rhs={self.rhs}, number={self.number})"


class MetaSyntaxAST:
    """
    The abstract syntax tree of the meta-syntax.
    """

    def __init__(self, rules: List[Rule]) -> None:
        """
        Initializes the MetaSyntaxAST.

        :param rules: A list of rules.
        """
        self.rules: List[Rule] = rules

    def __rich__(self) -> Tree:
        tree = Tree(f"[cyan]Rules[/cyan]")
        for rule in self.rules:
            tree.add(rule)
        return tree

    def __repr__(self) -> str:
        return f"MetaSyntaxAST(rules={self.rules})"


Operands = Union[Terminal, NonTerminal, Conjunction, Disjunction]
PostfixCapable = Union[NonTerminal, Conjunction, Disjunction]
