from typing import List, Union

from rich.tree import Tree


class Variable:
    """
    Represents a variable in the Simply Typed Lambda Calculus (STLC).

    In STLC, variables are typed and can be bound by abstractions
    (lambda terms). They are the basic building blocks of lambda terms and
    can represent both free and bound variables in expressions.
    """

    def __init__(self, name: str) -> None:
        """
        Initializes a variable with a given name.

        :param name: The name of the variable, typically a single letter
                     in STLC.
        """
        self.name: str = name

    def __rich__(self) -> Tree:
        return Tree(f"[blue]Variable[/blue]: [white]{self.name}[/white]")

    def __repr__(self) -> str:
        return f"Variable[Name: {self.name}]"


class Application:
    """
    Represents function application in STLC.

    Application is one of the two main constructs in lambda calculus,
    alongside abstraction. It represents the act of applying a function
    (the left term) to an argument (the right term). In STLC, the types of
    the function and argument must be compatible for the application to
    be valid.
    """

    def __init__(self, function: "Expression", argument: "Expression") -> None:
        """
        Initializes an application with a function and its argument.

        :param function: The function (left term) of the application.
        :param argument: The argument (right term) to which the function is applied.
        """
        self.function: Expression = function
        self.argument: Expression = argument

    def __rich__(self) -> Tree:
        tree = Tree(f"[bright_yellow]Application[/bright_yellow]")
        function_tree = Tree(f"[yellow]Function[/yellow]")
        function_tree.add(self.function)
        argument_tree = Tree(f"[yellow]Argument[/yellow]")
        argument_tree.add(self.argument)
        tree.add(function_tree)
        tree.add(argument_tree)
        return tree

    def __repr__(self) -> str:
        return f"Application[]"


class Abstraction:
    """
    Represents a lambda abstraction in STLC.

    Abstraction is a core concept in lambda calculus, introducing a bound
    variable and a body in which that variable may appear. In STLC,
    abstractions are typed, meaning the bound variable has an explicit type
    annotation. This is a key difference from untyped lambda calculus.
    """

    def __init__(
        self, variable: Variable, type: "Type", expression: "Expression"
    ) -> None:
        """
        Initializes a lambda abstraction.

        :param variable: The bound variable of the abstraction.
        :param type: The type of the bound variable, a key feature of STLC.
        :param expression: The body of the abstraction, an expression that
                           may contain the bound variable.
        """
        self.variable: Variable = variable
        self.type: Type = type
        self.expression: Expression = expression

    def __rich__(self) -> Tree:
        tree = Tree(f"[bright_red]Abstraction[/bright_red]: [white]λ{self.variable.name}[/white]")
        tree.add(self.type)
        tree.add(self.expression)
        return tree

    def __repr__(self) -> str:
        return f"Abstraction[Variable: {self.variable.name}]"


class SimpleType:
    """
    Represents a simple type in STLC.

    Types are a fundamental aspect of STLC, distinguishing it from untyped
    lambda calculus. Simple types can represent basic types (like Int, Bool)
    or type variables. They are used to ensure type consistency and enable
    type checking in STLC expressions.
    """

    def __init__(self, name: str) -> None:
        """
        Initializes a simple type.

        :param name: The name of the type.
        """
        self.name: str = name

    def __rich__(self) -> Tree:
        return Tree(f"[green]Simple Type[/green]: [white]{self.name}[/white]")

    def __repr__(self) -> str:
        return f"SimpleType[Name: {self.name}]"


class ArrowType:
    """
    Represents a function type (arrow type) in STLC.

    Arrow types are compound types used to represent functions. They consist
    of an input type (antecedent) and an output type (consequent). In STLC,
    all functions are unary (single-argument), so multi-argument functions
    are represented through currying.
    """

    def __init__(self, antecedent: "Type", consequent: "Type") -> None:
        """
        Initializes an arrow type.

        :param antecedent: The input type of the function.
        :param consequent: The output type of the function.
        """
        self.antecedent: Type = antecedent
        self.consequent: Type = consequent

    def __rich__(self) -> Tree:
        tree = Tree(f"[red]Arrow Type[/red]: [white]→[/white]")
        antecedent_tree = Tree("Antecedent")
        antecedent_tree.add(self.antecedent)
        tree.add(antecedent_tree)
        consequent_tree = Tree("Consequent")
        consequent_tree.add(self.consequent)
        tree.add(consequent_tree)
        return tree

    def __repr__(self) -> str:
        return "ArrowType[<>]"


class Declaration:
    """
    Represents a type declaration in STLC.

    Declarations associate expressions with their types. They are used in
    contexts to specify the types of free variables or to define typed
    constants. In STLC, all expressions must be well-typed according to these
    declarations.
    """

    def __init__(self, expression: "Expression", type: "Type") -> None:
        """
        Initializes a declaration.

        :param expression: The expression being declared.
        :param type: The type assigned to the expression.
        """
        self.expression: Expression = expression
        self.type: Type = type

    def __rich__(self) -> Tree:
        tree = Tree(f"[magenta]Declaration[/magenta]")
        tree.add(self.expression)
        tree.add(self.type)
        return tree

    def __repr__(self) -> str:
        return f"Declaration[<>]"


class Statement:
    """
    Represents a typing statement in STLC.

    A statement asserts that an expression has a particular type. In the
    context of type checking or inference, statements are what we aim to
    prove or derive based on the given declarations and typing rules of STLC.
    """

    def __init__(self, expression: "Expression", type: "Type") -> None:
        """
        Initializes a typing statement.

        :param expression: The expression whose type is being stated.
        :param type: The type asserted for the expression.
        """
        self.expression: Expression = expression
        self.type: Type = type

    def __rich__(self) -> Tree:
        tree = Tree(f"[yellow]Statement[/yellow]: [white]Λ[/white]")
        tree.add(self.expression)
        tree.add(self.type)
        return tree

    def __repr__(self) -> str:
        return f"Statement[<>]"


class Judgement:
    """
    Represents a typing judgement in STLC.

    A judgement consists of a typing context (a set of assumptions about types
    of variables) and a statement to be proved. It forms the basis of type
    checking and inference in STLC.
    """

    def __init__(self, context: List[Declaration], statement: "Statement") -> None:
        """
        Initializes a typing judgement.

        :param context: A list of declarations forming the typing context.
        :param statement: The typing statement to be proved under the given
                          context.
        """
        self.context: List[Declaration] = context if context else []
        self.statement: Statement = statement

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

    def __repr__(self) -> str:
        return f"Judgement[Context Count: {len(self.context)}]"


class AST:
    """
    Represents the Abstract Syntax Tree (AST) of an STLC program.

    The AST captures the structure of the STLC program, consisting of a
    series of typing judgements. It serves as the foundation for type
    checking, evaluation, and other analyses of STLC programs.
    """

    def __init__(self, judgements: List[Judgement]) -> None:
        """
        Initializes the AST with a list of typing judgements.

        :param judgements: A list of typing judgements that make up the
                           STLC program.
        """
        self.judgements: List[Judgement] = judgements

    def __rich__(self) -> Tree:
        tree = Tree(f"[cyan]Abstract Syntax Tree[/cyan]")
        for judgement in self.judgements:
            tree.add(judgement)
        return tree

    def __repr__(self) -> str:
        return f"AST[Judgement Count: {len(self.judgements)}]"


Expression = Union[Application, Abstraction, Variable]
Type = Union[SimpleType, ArrowType]
