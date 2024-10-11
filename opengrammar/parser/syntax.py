from enum import Enum
from typing import List, Union

from rich.tree import Tree


class Name:
    """
    The name of a constant.
    """

    def __init__(self, name: str) -> None:
        """
        Initializes the name.

        :param name: The name of the constant.
        """
        self.name: str = name

    def __repr__(self) -> str:
        return f"Name(name='{self.name}')"

    def __rich__(self) -> Tree:
        return Tree(f"[blue]Name[/blue]: [white]{self.name}[/white]")


class Variable:
    """
    The name of a variable.
    """

    def __init__(self, name: str) -> None:
        """
        Initializes the name.

        :param name: The name of the variable.
        """
        self.name: str = name

    def __repr__(self) -> str:
        return f"Variable(name='{self.name}')"

    def __rich__(self) -> Tree:
        return Tree(f"[blue]Variable[/blue]: [white]{self.name}[/white]")


class SimpleType:
    """
    The name of a type.
    """

    def __init__(self, name: str) -> None:
        """
        Initializes the name.

        :param name: The name of the type.
        """
        self.name: str = name

    def __repr__(self) -> str:
        return f"SimpleType(name='{self.name}')"

    def __rich__(self) -> Tree:
        return Tree(f"[green]Simple Type[/green]: [white]{self.name}[/white]")


class BiConditional:
    """
    The BiConditional Connective.
    """

    def __init__(self, antecedent: "Expression", consequent: "Expression") -> None:
        """
        Initializes the BiConditional.

        :param antecedent: The antecedent of the BiConditional.
        :param consequent: The consequent of the BiConditional.
        """
        self.antecedent: Expression = antecedent
        self.consequent: Expression = consequent

    def __repr__(self) -> str:
        return (
            f"BiConditional(antecedent={self.antecedent}, consequent={self.consequent})"
        )

    def __rich__(self) -> Tree:
        tree = Tree(f"[bright_cyan]BiConditional[/bright_cyan]: [white]↔[/white]")
        tree.add(self.antecedent)
        tree.add(self.consequent)
        return tree


class Conditional:
    """
    The Conditional Connective.
    """

    def __init__(self, antecedent: "Expression", consequent: "Expression") -> None:
        """
        Initializes the Conditional.

        :param antecedent: The antecedent of the Conditional.
        :param consequent: The consequent of the Conditional.
        """
        self.antecedent: Expression = antecedent
        self.consequent: Expression = consequent

    def __repr__(self) -> str:
        return (
            f"Conditional(antecedent={self.antecedent}, consequent={self.consequent})"
        )

    def __rich__(self) -> Tree:
        tree = Tree(f"[bright_green]Conditional[/bright_green]: [white]→[/white]")
        tree.add(self.antecedent)
        tree.add(self.consequent)
        return tree


class Disjunction:
    """
    The Disjunction Connective.
    """

    def __init__(self, antecedent: "Expression", consequent: "Expression") -> None:
        """
        Initializes the Disjunction.

        :param antecedent: The antecedent of the Disjunction.
        :param consequent: The consequent of the Disjunction.
        """
        self.antecedent: Expression = antecedent
        self.consequent: Expression = consequent

    def __repr__(self) -> str:
        return (
            f"Disjunction(antecedent={self.antecedent}, consequent={self.consequent})"
        )

    def __rich__(self) -> Tree:
        tree = Tree(f"[bright_yellow]Disjunction[/bright_yellow]: [white]∨[/white]")
        tree.add(self.antecedent)
        tree.add(self.consequent)
        return tree


class Conjunction:
    """
    The Conjunction Connective.
    """

    def __init__(self, antecedent: "Expression", consequent: "Expression") -> None:
        """
        Initializes the Conjunction.

        :param antecedent: The antecedent of the Conjunction.
        :param consequent: The consequent of the Conjunction.
        """
        self.antecedent: Expression = antecedent
        self.consequent: Expression = consequent

    def __repr__(self) -> str:
        return (
            f"Conjunction(antecedent={self.antecedent}, consequent={self.consequent})"
        )

    def __rich__(self) -> Tree:
        tree = Tree(f"[bright_magenta]Conjunction[/bright_magenta]: [white]∧[/white]")
        tree.add(self.antecedent)
        tree.add(self.consequent)
        return tree


class Negation:
    """
    A unary negation connective.
    """

    def __init__(self, expression: "Expression") -> None:
        """
        Initializes the Negation.

        :param expression: The expression to be negated.
        """
        self.expression: Expression = expression

    def __repr__(self) -> str:
        return f"Negation(expression={self.expression})"

    def __rich__(self) -> Tree:
        tree = Tree(f"[bright_red]Negation[/bright_red]: [white]¬[/white]")
        tree.add(self.expression)
        return tree


class Equality:
    """
    The Equality Predicate.
    """

    def __init__(self, antecedent: "Term", consequent: "Term") -> None:
        """
        Initializes the Equality.

        :param antecedent: The antecedent of the Equality.
        :param consequent: The consequent of the Equality.
        """
        self.antecedent: Term = antecedent
        self.consequent: Term = consequent

    def __repr__(self) -> str:
        return f"Equality(antecedent={self.antecedent}, consequent={self.consequent})"

    def __rich__(self) -> Tree:
        tree = Tree(f"[bright_white]Equality[/bright_white]: [white]=[/white]")
        tree.add(self.antecedent)
        tree.add(self.consequent)
        return tree


class Predicate:
    """
    The Predicate.
    """

    def __init__(self, name: str, terms: List["Term"]) -> None:
        """
        Initializes the Predicate.

        :param name: The name of the predicate.
        :param terms: The terms of the predicate.
        """
        self.name: str = name
        self.terms: List[Term] = terms

    def __repr__(self) -> str:
        return f"Predicate(name='{self.name}', terms={self.terms})"

    def __rich__(self) -> Tree:
        tree = Tree(f"[cyan]Predicate[/cyan]: [white]{self.name}[/white]")
        for term in self.terms:
            tree.add(term)
        return tree


class QuantifierOperator(Enum):
    UNIVERSAL = "∀"
    EXISTENTIAL = "∃"


class QuantifiedExpression:
    """
    A quantified expression.
    """

    def __init__(
        self,
        quantifier: QuantifierOperator,
        variable: Variable,
        type_expression: "Type",
        expression: "Expression",
    ) -> None:
        """
        Initializes the QuantifiedExpression.

        :param quantifier: Either UNIVERSAL or EXISTENTIAL.
        :param variable: The variable in the quantified expression.
        :param type_expression: The type of the variable.
        :param expression: The expression in the quantified expression.
        """
        self.quantifier: QuantifierOperator = quantifier
        self.variable: Variable = variable
        self.type_expression: "Type" = type_expression
        self.expression: Expression = expression

    def __repr__(self) -> str:
        return f"QuantifiedExpression(quantifier={self.quantifier}, variable={self.variable}, type_expression={self.type_expression}, expression={self.expression})"

    def __rich__(self) -> Tree:
        tree = Tree(
            f"[bright_red]Quantified Expression[/bright_red]: [white]{self.quantifier.value}[/white]"
        )
        tree.add(self.variable.__rich__().add(self.type_expression))
        tree.add(self.expression)
        return tree


class TypeBiConditional:
    """
    A bi-conditional type.
    """

    def __init__(self, antecedent: "Type", consequent: "Type") -> None:
        """
        Initializes the TypeBiConditional.

        :param antecedent: The antecedent type.
        :param consequent: The consequent type.
        """
        self.antecedent: "Type" = antecedent
        self.consequent: "Type" = consequent

    def __repr__(self) -> str:
        return f"TypeBiConditional(antecedent={self.antecedent}, consequent={self.consequent})"

    def __rich__(self) -> Tree:
        tree = Tree(f"[bright_cyan]Type BiConditional[/bright_cyan]: [white]↔[/white]")
        tree.add(self.antecedent)
        tree.add(self.consequent)
        return tree


class TypeConditional:
    """
    A conditional type.
    """

    def __init__(self, antecedent: "Type", consequent: "Type") -> None:
        """
        Initializes the TypeConditional.

        :param antecedent: The antecedent type.
        :param consequent: The consequent type.
        """
        self.antecedent: "Type" = antecedent
        self.consequent: "Type" = consequent

    def __repr__(self) -> str:
        return f"TypeConditional(antecedent={self.antecedent}, consequent={self.consequent})"

    def __rich__(self) -> Tree:
        tree = Tree(f"[bright_green]Type Conditional[/bright_green]: [white]→[/white]")
        tree.add(self.antecedent)
        tree.add(self.consequent)
        return tree


class TypeDisjunction:
    """
    A disjunction type.
    """

    def __init__(self, antecedent: "Type", consequent: "Type") -> None:
        """
        Initializes the TypeDisjunction.

        :param antecedent: The antecedent type.
        :param consequent: The consequent type.
        """
        self.antecedent: "Type" = antecedent
        self.consequent: "Type" = consequent

    def __repr__(self) -> str:
        return f"TypeDisjunction(antecedent={self.antecedent}, consequent={self.consequent})"

    def __rich__(self) -> Tree:
        tree = Tree(
            f"[bright_yellow]Type Disjunction[/bright_yellow]: [white]∨[/white]"
        )
        tree.add(self.antecedent)
        tree.add(self.consequent)
        return tree


class TypeConjunction:
    """
    A conjunction type.
    """

    def __init__(self, antecedent: "Type", consequent: "Type") -> None:
        """
        Initializes the TypeConjunction.

        :param antecedent: The antecedent type.
        :param consequent: The consequent type.
        """
        self.antecedent: "Type" = antecedent
        self.consequent: "Type" = consequent

    def __repr__(self) -> str:
        return f"TypeConjunction(antecedent={self.antecedent}, consequent={self.consequent})"

    def __rich__(self) -> Tree:
        tree = Tree(
            f"[bright_magenta]Type Conjunction[/bright_magenta]: [white]∧[/white]"
        )
        tree.add(self.antecedent)
        tree.add(self.consequent)
        return tree


class Declaration:
    """
    A declaration.
    """

    def __init__(self, term: Union["Term", "Type"], type_expression: "Type") -> None:
        """
        Initializes the Declaration.

        :param term: The term being declared.
        :param type_expression: The type of the term being declared.
        """
        self.term: Union[Term, Type] = term
        self.type_expression: Type = type_expression

    def __repr__(self) -> str:
        return f"Declaration(term={self.term}, type_expression={self.type_expression})"

    def __rich__(self) -> Tree:
        tree = Tree(f"[magenta]Declaration[/magenta]")
        inhabitant = Tree(f"[bright_blue]Inhabitant[/bright_blue]")
        inhabitant.add(self.term)
        type = Tree(f"[bright_cyan]Type[/bright_cyan]")
        type.add(self.type_expression)
        tree.add(inhabitant)
        tree.add(type)
        return tree


class Statement:
    """
    A statement.
    """

    def __init__(self, expression: "Expression") -> None:
        """
        Initializes the Statement.

        :param expression: The expression being declared.
        """
        self.expression: Expression = expression

    def __repr__(self) -> str:
        return f"Statement(expression={self.expression})"

    def __rich__(self) -> Tree:
        tree = Tree(f"[yellow]Statement[/yellow]")
        tree.add(self.expression)
        return tree


class TypeStatement:
    """
    A type statement.
    """

    def __init__(self, term: "Term", type_expression: "Type") -> None:
        """
        Initializes the TypeStatement.

        :param term: The term being declared.
        :param type_expression: The type of the term being declared.
        """
        self.term: Term = term
        self.type_expression: "Type" = type_expression

    def __repr__(self) -> str:
        return (
            f"TypeStatement(term={self.term}, type_expression={self.type_expression})"
        )

    def __rich__(self) -> Tree:
        tree = Tree(f"[yellow]Type Statement[/yellow]")
        tree.add(self.term)
        tree.add(self.type_expression)
        return tree


class Judgement:
    """
    A judgement.
    """

    def __init__(self, context: List[Declaration], statement: Statement) -> None:
        """
        Initializes the Judgement.

        :param context: The context of the judgement.
        :param statement: The statement being judged.
        """
        self.context: List[Declaration] = context
        self.statement: Statement = statement

    def __repr__(self) -> str:
        return f"Judgement(context={self.context}, statement={self.statement})"

    def __rich__(self) -> Tree:
        tree = Tree(f"[bright_magenta]Judgement[/bright_magenta]")
        if self.context:
            context_tree = Tree(f"[yellow]Context[/yellow]: [white]Γ[/white]")
            for declaration in self.context:
                context_tree.add(declaration)
            tree.add(context_tree)
        else:
            context_tree = Tree(f"[yellow]Empty Context[/yellow]: [white]θ[/white]")
            tree.add(context_tree)
        turnstile_tree = Tree(f"[yellow]Derivable[/yellow]: [white]⊢[/white]")
        turnstile_tree.add(self.statement)
        tree.add(turnstile_tree)
        return tree


class AST:
    """
    Represents the Abstract Syntax Tree (AST) of an OpenGrammar program.

    The AST captures the structure of the OpenGrammar program, consisting of a
    series of typing judgements. It serves as the foundation for type
    checking, evaluation, and other analyses of OpenGrammar programs.
    """

    def __init__(self, judgements: List[Judgement]) -> None:
        """
        Initializes the AST with a list of typing judgements.

        :param judgements: A list of typing judgements that make up the
                           OpenGrammar program.
        """
        self.judgements: List[Judgement] = judgements

    def __rich__(self) -> Tree:
        tree = Tree(f"[cyan]Abstract Syntax Tree[/cyan]")
        for judgement in self.judgements:
            tree.add(judgement)
        return tree

    def __repr__(self) -> str:
        return f"AST[Judgement Count: {len(self.judgements)}]"


Term = Union[Variable, str]
Atom = Union[Predicate, QuantifiedExpression, Equality, TypeStatement]
Expression = Union[
    BiConditional,
    Conditional,
    Disjunction,
    Conjunction,
    Negation,
    Atom,
]
Type = Union[
    TypeBiConditional, TypeConditional, TypeDisjunction, TypeConjunction, SimpleType
]
