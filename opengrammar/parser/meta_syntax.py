"""
The meta-syntax parser.
"""

from typing import List, Optional, Union

from rich.tree import Tree


class NonTerminal:
    """
    A non-terminal symbol.
    """

    def __init__(self, symbol: str, number: Optional[int] = None) -> None:
        """
        Initializes the NonTerminal.

        :param symbol: The symbol of the non-terminal.
        :param number: The number of the non-terminal.
        """
        self.symbol: str = symbol
        self.number: Optional[int] = number

    def __rich__(self) -> str:
        if self.number:
            string = (
                f"[bright_green]NonTerminal[/bright_green] "
                f"[[yellow]{self.number}[/yellow]]: "
                f"[bright_blue]{self.symbol}[/bright_blue]"
            )
        else:
            string = (
                f"[bright_green]NonTerminal[/bright_green]: "
                f"[bright_blue]{self.symbol}[/bright_blue]"
            )
        return string

    def __repr__(self) -> str:
        return f"NonTerminal(symbol={self.symbol}, number={self.number})"


class Terminal:
    """
    A terminal symbol.
    """

    def __init__(self, symbol: str, number: Optional[int] = None) -> None:
        """
        Initializes the Terminal.

        :param symbol: The symbol of the terminal.
        :param number: The number of the terminal.
        """
        self.symbol: str = symbol
        self.number: Optional[int] = number

    def __rich__(self) -> str:
        if self.number:
            string = (
                f"[bright_red]Terminal[/bright_red]    "
                f"[[yellow]{self.number}[/yellow]]: "
                f"[bright_blue]{self.symbol}[/bright_blue]"
            )
        else:
            string = (
                f"[bright_red]Terminal[/bright_red]   : "
                f"[bright_blue]{self.symbol}[/bright_blue]"
            )
        return string

    def __repr__(self) -> str:
        return f"Terminal(symbol={self.symbol}, number={self.number})"


class LHS:
    """
    The left-hand side of a rule.
    """

    def __init__(self, rules: List[Union[NonTerminal, Terminal]]):
        """
        Initializes the LHS.

        :param rules: A list of terminals and non-terminals.
        """
        self.rules: List[Union[NonTerminal, Terminal]] = rules

    def __rich__(self) -> Tree:
        tree = Tree("LHS")
        for rule in self.rules:
            tree.add(rule)
        return tree

    def __repr__(self) -> str:
        return f"LHS(rules={self.rules})"


class RHS:
    """
    The right-hand side of a rule.
    """

    def __init__(self, rules: List[Union[NonTerminal, Terminal]]):
        """
        Initializes the RHS.

        :param rules: A list of terminals and non-terminals.
        """
        self.rules: List[Union[NonTerminal, Terminal]] = rules

    def __rich__(self) -> Tree:
        tree = Tree("RHS")
        for rule in self.rules:
            tree.add(rule)
        return tree

    def __repr__(self) -> str:
        return f"RHS(rules={self.rules})"


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

    def __repr__(self):
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
