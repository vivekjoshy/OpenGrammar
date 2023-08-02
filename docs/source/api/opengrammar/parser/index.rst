:py:mod:`opengrammar.parser`
============================

.. py:module:: opengrammar.parser

.. autoapi-nested-parse::

   The parser module contains the parser for the Meta Syntax and the Universal Grammar.



Submodules
----------
.. toctree::
   :titlesonly:
   :maxdepth: 1

   meta_syntax/index.rst
   transformer/index.rst


Package Contents
----------------

Classes
~~~~~~~

.. autoapisummary::

   opengrammar.parser.MetaSyntaxParser
   opengrammar.parser.UniversalParser




.. py:class:: MetaSyntaxParser


   Parses Meta Syntax into an AST.

   Initializes the MetaSyntaxParser.

   .. py:method:: parse(text)

      Parses the text into an AST.

      :param text: A string with valid Meta Syntax.
      :return: A Meta Syntax AST.



.. py:class:: UniversalParser(grammar)


   Parses a Universal Grammar into an AST.

   Initializes the UniversalParser.

   :param grammar: A valid Universal Grammar string.


