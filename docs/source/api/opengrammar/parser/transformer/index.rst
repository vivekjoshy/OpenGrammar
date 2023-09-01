:py:mod:`opengrammar.parser.transformer`
========================================

.. py:module:: opengrammar.parser.transformer

.. autoapi-nested-parse::

   This module contains the MetaSyntaxTransformer class, which is used to
   transform the Lark parse tree into a MetaSyntaxAST.



Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   opengrammar.parser.transformer.MetaSyntaxTransformer




.. py:class:: MetaSyntaxTransformer(visit_tokens = True)


   Bases: :py:obj:`lark.Transformer`\ [\ :py:obj:`lark.Token`\ , :py:obj:`opengrammar.parser.meta_syntax.MetaSyntaxAST`\ ]

   Transforms the Lark parse tree into a MetaSyntaxAST.

   .. py:method:: NEWLINE(token)

      Discards newlines.

      :param token: A newline token.


   .. py:method:: NON_TERMINAL(string)

      Creates a non-terminal from a string.

      :param string: A string without quotes.


   .. py:method:: OR_OPERATOR(token)

      Discards OR tokens.

      :param token: An OR token.


   .. py:method:: TERMINAL(string)

      Creates a terminal from a string.

      :param string: The string with quotes.


   .. py:method:: WS(token)

      Discards whitespace.

      :param token: A whitespace token.


   .. py:method:: lhs(children)

      Creates an LHS from a list of terminals and non-terminals.

      :param children: A list of terminals and non-terminals.


   .. py:method:: lines(children)

      Creates a MetaSyntaxAST from a list of rules.

      :param children: A list of rules.


   .. py:method:: non_terminal(children)

      Returns the first child of the non-terminal.

      :param children: A list of non-terminals.


   .. py:method:: rhs(children)

      Creates an RHS from a list of terminals and non-terminals.

      :param children: A list of terminals and non-terminals.


   .. py:method:: rule(children)

      Creates a rule from an LHS and RHS.

      :param children: A list of LHS and RHS.


   .. py:method:: separator(token)

      Discards separators.

      :param token: A separator token.


   .. py:method:: terminal(children)

      Returns the first child of the terminal.

      :param children: A list of terminals.



