:py:mod:`opengrammar.parser.meta_syntax`
========================================

.. py:module:: opengrammar.parser.meta_syntax

.. autoapi-nested-parse::

   The meta-syntax parser.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   opengrammar.parser.meta_syntax.LHS
   opengrammar.parser.meta_syntax.MetaSyntaxAST
   opengrammar.parser.meta_syntax.NonTerminal
   opengrammar.parser.meta_syntax.RHS
   opengrammar.parser.meta_syntax.Rule
   opengrammar.parser.meta_syntax.Terminal




.. py:class:: LHS(rules)


   The left-hand side of a rule.

   Initializes the LHS.

   :param rules: A list of terminals and non-terminals.


.. py:class:: MetaSyntaxAST(rules)


   The abstract syntax tree of the meta-syntax.

   Initializes the MetaSyntaxAST.

   :param rules: A list of rules.


.. py:class:: NonTerminal(symbol, number = None)


   A non-terminal symbol.

   Initializes the NonTerminal.

   :param symbol: The symbol of the non-terminal.
   :param number: The number of the non-terminal.


.. py:class:: RHS(rules)


   The right-hand side of a rule.

   Initializes the RHS.

   :param rules: A list of terminals and non-terminals.


.. py:class:: Rule(lhs, rhs, number = None)


   A rule.

   Initializes the Rule.

   :param lhs: The left-hand side of the rule.
   :param rhs: The right-hand side of the rule.
   :param number: The rule number.


.. py:class:: Terminal(symbol, number = None)


   A terminal symbol.

   Initializes the Terminal.

   :param symbol: The symbol of the terminal.
   :param number: The number of the terminal.


