"""
Safe Pratt parser (precedence climbing) for mathematical expressions in MathField.
Guarantees absolute security with zero use of eval, exec, or ast.literal_eval.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from mathfield.core.ast.nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    FunctionCallNode,
    FUNCTION_ARITIES,
    ParameterNode,
    UnaryOpNode,
    VariableNode,
)
from mathfield.core.ast.tokens import Token, TokenType, tokenize


class ParseError(ValueError):
    """
    Exception raised when expression parsing fails, with detailed source context.
    """
    def __init__(
        self,
        message: str,
        token: Optional[Token] = None,
        source_text: str = "",
        line: Optional[int] = None,
        column: Optional[int] = None,
    ):
        self.message = message
        self.token = token
        self.source_text = source_text
        self.line = token.line if token else (line or 1)
        self.column = token.column if token else (column or 1)

        lines = source_text.splitlines() if source_text else []
        line_str = lines[self.line - 1] if 0 <= self.line - 1 < len(lines) else ""
        pointer = " " * max(0, self.column - 1) + "^"
        formatted = (
            f"ParseError at line {self.line}, column {self.column}:\n"
            f"  {line_str}\n"
            f"  {pointer}\n"
            f"{message}"
        )
        super().__init__(formatted)


# Infix binding powers: (left_binding_power, right_binding_power)
# Left-associative: l_bp < r_bp (e.g. 10, 11)
# Right-associative: l_bp > r_bp (e.g. 31, 30)
INFIX_BINDING_POWERS: dict[TokenType, Tuple[int, int]] = {
    TokenType.PLUS: (10, 11),
    TokenType.MINUS: (10, 11),
    TokenType.STAR: (20, 21),
    TokenType.SLASH: (20, 21),
    TokenType.PERCENT: (20, 21),
    TokenType.CARET: (31, 30),
}

PREFIX_BINDING_POWER = 25


class Parser:
    """
    O(N) recursive precedence-climbing (Pratt) parser for mathematical expressions.
    """

    def __init__(self, tokens: List[Token], source_text: str = ""):
        self.tokens = tokens
        self.source_text = source_text
        self.cursor = 0

    def at_end(self) -> bool:
        return self.cursor >= len(self.tokens) or self.tokens[self.cursor].type == TokenType.EOF

    def current(self) -> Token:
        if self.cursor < len(self.tokens):
            return self.tokens[self.cursor]
        return self.tokens[-1]

    def peek(self, offset: int = 1) -> Token:
        pos = self.cursor + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return self.tokens[-1]

    def consume(self) -> Token:
        tok = self.current()
        if not self.at_end():
            self.cursor += 1
        return tok

    def match(self, token_type: TokenType) -> bool:
        if self.current().type == token_type:
            self.consume()
            return True
        return False

    def expect(self, token_type: TokenType, err_msg: Optional[str] = None) -> Token:
        tok = self.current()
        if tok.type != token_type:
            msg = err_msg or f"Expected {token_type.name}, found {tok.raw_str!r}"
            raise ParseError(msg, token=tok, source_text=self.source_text)
        return self.consume()

    def parse(self) -> ASTNode:
        """
        Parses the full token stream into an ASTNode.
        Ensures no trailing unparsed tokens remain.
        """
        if self.at_end():
            raise ParseError("Empty expression", token=self.current(), source_text=self.source_text)

        expr = self.parse_expression(min_bp=0)

        if not self.at_end():
            tok = self.current()
            raise ParseError(
                f"Unexpected trailing token: {tok.raw_str!r}",
                token=tok,
                source_text=self.source_text,
            )

        return expr

    def parse_expression(self, min_bp: int = 0) -> ASTNode:
        """
        Precedence-climbing recursive parsing step.
        """
        if self.at_end():
            raise ParseError("Unexpected end of expression", token=self.current(), source_text=self.source_text)

        token = self.consume()
        lhs = self._parse_nud(token)

        while not self.at_end():
            next_tok = self.current()

            # Delimiters that end subexpressions
            if next_tok.type in (TokenType.RPAREN, TokenType.COMMA, TokenType.EOF):
                break

            bp = INFIX_BINDING_POWERS.get(next_tok.type)
            if bp is None:
                raise ParseError(
                    f"Unexpected token {next_tok.raw_str!r} where operator was expected",
                    token=next_tok,
                    source_text=self.source_text,
                )

            l_bp, r_bp = bp
            if l_bp < min_bp:
                break

            op_tok = self.consume()
            rhs = self.parse_expression(r_bp)

            op_sym = "^" if op_tok.type == TokenType.CARET else op_tok.raw_str
            lhs = BinaryOpNode(op_sym, lhs, rhs)

        return lhs

    def _parse_nud(self, token: Token) -> ASTNode:
        """
        Null Denotation: dispatches on tokens appearing at the beginning of an operand.
        """
        # 1. Literals & Constants
        if token.type == TokenType.NUMBER:
            return ConstantNode(token.value)
        elif token.type == TokenType.CONSTANT:
            return ConstantNode(token.value, symbol=token.raw_str.lower())
        elif token.type == TokenType.VARIABLE:
            return VariableNode(token.value)
        elif token.type == TokenType.PARAMETER:
            param_name, param_index = token.value
            return ParameterNode(param_name, param_index)

        # 2. Parenthesized expressions
        elif token.type == TokenType.LPAREN:
            inner_expr = self.parse_expression(0)
            self.expect(TokenType.RPAREN, f"Unmatched '(' - missing closing ')'")
            return inner_expr

        # 3. Unary operators (+, -)
        elif token.type == TokenType.PLUS:
            operand = self.parse_expression(PREFIX_BINDING_POWER)
            return UnaryOpNode("+", operand)
        elif token.type == TokenType.MINUS:
            operand = self.parse_expression(PREFIX_BINDING_POWER)
            return UnaryOpNode("-", operand)

        # 4. Function call
        elif token.type == TokenType.IDENTIFIER:
            func_name = token.value.lower()
            if func_name not in FUNCTION_ARITIES:
                raise ParseError(
                    f"Disallowed or unknown function {token.raw_str!r}. "
                    f"Allowed functions: {sorted(list(FUNCTION_ARITIES.keys()))}",
                    token=token,
                    source_text=self.source_text,
                )

            self.expect(TokenType.LPAREN, f"Expected '(' after function name {func_name!r}")
            args: List[ASTNode] = []
            if self.current().type != TokenType.RPAREN:
                args.append(self.parse_expression(0))
                while self.match(TokenType.COMMA):
                    args.append(self.parse_expression(0))

            self.expect(TokenType.RPAREN, f"Expected ')' closing arguments for function {func_name!r}")

            expected_arity = FUNCTION_ARITIES[func_name]
            if len(args) != expected_arity:
                raise ParseError(
                    f"Function {func_name!r} expects {expected_arity} arguments, but received {len(args)}",
                    token=token,
                    source_text=self.source_text,
                )

            return FunctionCallNode(func_name, tuple(args))

        raise ParseError(
            f"Unexpected token {token.raw_str!r} at start of expression",
            token=token,
            source_text=self.source_text,
        )


def parse(source_text: str, insert_implicit_mul: bool = True) -> ASTNode:
    """
    Parses a mathematical expression string into an ASTNode.

    Parameters:
        source_text: The mathematical formula as a string.
        insert_implicit_mul: Whether to automatically resolve implicit multiplication.

    Returns:
        The root ASTNode of the expression tree.
    """
    tokens = tokenize(source_text, insert_implicit_mul=insert_implicit_mul)
    parser = Parser(tokens, source_text=source_text)
    return parser.parse()


class PrattParser:
    """Wrapper class providing object-oriented Pratt parsing interface."""

    def __init__(self, insert_implicit_mul: bool = True) -> None:
        self.insert_implicit_mul = insert_implicit_mul

    def parse(self, source_text: str) -> ASTNode:
        return parse(source_text, insert_implicit_mul=self.insert_implicit_mul)

