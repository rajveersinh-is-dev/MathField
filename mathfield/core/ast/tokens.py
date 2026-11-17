"""
Token definitions and lexical analysis for the MathField expression language.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, List


class TokenType(Enum):
    """Enumeration of all lexical token types in MathField expressions."""

    # Literals
    NUMBER = auto()  # e.g. 42, 3.14159, 1e-4, .5

    # Mathematical Constants
    CONSTANT = auto()  # pi, e

    # Coordinates & Parameters
    VARIABLE = auto()  # x, y, u, v
    PARAMETER = auto()  # c0..cN, theta0..thetaN, p0..pN

    # Operators
    PLUS = auto()  # +
    MINUS = auto()  # -
    STAR = auto()  # *
    SLASH = auto()  # /
    CARET = auto()  # ^ or **
    PERCENT = auto()  # %

    # Delimiters
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    COMMA = auto()  # ,

    # Identifiers (for whitelisted functions)
    IDENTIFIER = auto()  # sin, cos, rbf_gaussian, etc.

    # Control
    EOF = auto()  # End of expression


@dataclass(frozen=True, slots=True)
class Token:
    """
    Immutable representation of a lexical token with source position tracking.
    """

    type: TokenType
    value: Any
    raw_str: str
    line: int
    column: int
    length: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, line={self.line}, col={self.column})"


class LexerError(ValueError):
    """Exception raised when lexical scanning encounters invalid input."""

    def __init__(self, message: str, line: int, column: int, source_text: str = ""):
        self.message = message
        self.line = line
        self.column = column
        self.source_text = source_text

        lines = source_text.splitlines() if source_text else []
        line_str = lines[line - 1] if 0 <= line - 1 < len(lines) else ""
        pointer = " " * max(0, column - 1) + "^"
        formatted = f"LexerError at line {line}, column {column}:\n  {line_str}\n  {pointer}\n{message}"
        super().__init__(formatted)


# Parameter regex: c0..cN, theta0..thetaN, p0..pN
_PARAM_PATTERN = re.compile(r"^(c|theta|p)(\d+)$")

# Number regex: handles integer, float, leading dot, and scientific notation
_NUMBER_PATTERN = re.compile(r"^(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")

# Identifier regex
_IDENT_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*")

# Constant definitions
_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}

# Coordinate variable set
_VARIABLES = {"x", "y", "u", "v"}


def tokenize(source_text: str, insert_implicit_mul: bool = True) -> List[Token]:
    """
    Scans the source mathematical expression text into a list of Tokens.

    Parameters:
        source_text: Input string expression.
        insert_implicit_mul: Whether to insert implicit multiplication tokens
                             between adjacent factors (e.g. '2x' -> '2 * x').

    Returns:
        List of Token instances ending with TokenType.EOF.

    Raises:
        LexerError: If invalid characters or malformed tokens are encountered.
    """
    tokens: List[Token] = []
    length = len(source_text)
    idx = 0
    line = 1
    col = 1

    while idx < length:
        ch = source_text[idx]

        # Handle whitespace and line counting
        if ch == "\n":
            line += 1
            col = 1
            idx += 1
            continue
        elif ch in (" ", "\t", "\r"):
            col += 1
            idx += 1
            continue

        # Two-character operators: **
        if ch == "*" and idx + 1 < length and source_text[idx + 1] == "*":
            tokens.append(Token(TokenType.CARET, "^", "**", line, col, 2))
            idx += 2
            col += 2
            continue

        # Single-character operators and delimiters
        if ch == "+":
            tokens.append(Token(TokenType.PLUS, "+", "+", line, col, 1))
            idx += 1
            col += 1
            continue
        elif ch == "-":
            tokens.append(Token(TokenType.MINUS, "-", "-", line, col, 1))
            idx += 1
            col += 1
            continue
        elif ch == "*":
            tokens.append(Token(TokenType.STAR, "*", "*", line, col, 1))
            idx += 1
            col += 1
            continue
        elif ch == "/":
            tokens.append(Token(TokenType.SLASH, "/", "/", line, col, 1))
            idx += 1
            col += 1
            continue
        elif ch == "^":
            tokens.append(Token(TokenType.CARET, "^", "^", line, col, 1))
            idx += 1
            col += 1
            continue
        elif ch == "%":
            tokens.append(Token(TokenType.PERCENT, "%", "%", line, col, 1))
            idx += 1
            col += 1
            continue
        elif ch == "(":
            tokens.append(Token(TokenType.LPAREN, "(", "(", line, col, 1))
            idx += 1
            col += 1
            continue
        elif ch == ")":
            tokens.append(Token(TokenType.RPAREN, ")", ")", line, col, 1))
            idx += 1
            col += 1
            continue
        elif ch == ",":
            tokens.append(Token(TokenType.COMMA, ",", ",", line, col, 1))
            idx += 1
            col += 1
            continue

        # Numbers
        remainder = source_text[idx:]
        num_match = _NUMBER_PATTERN.match(remainder)
        if num_match:
            num_str = num_match.group(0)
            # Ensure a single dot is not matched as a number if it's just '.' without digits
            if num_str != ".":
                num_val = float(num_str)
                tok_len = len(num_str)
                tokens.append(Token(TokenType.NUMBER, num_val, num_str, line, col, tok_len))
                idx += tok_len
                col += tok_len
                continue

        # Identifiers, variables, parameters, constants
        ident_match = _IDENT_PATTERN.match(remainder)
        if ident_match:
            ident_str = ident_match.group(0)
            tok_len = len(ident_str)
            ident_lower = ident_str.lower()

            # Check mathematical constants
            if ident_lower in _CONSTANTS:
                tokens.append(Token(TokenType.CONSTANT, _CONSTANTS[ident_lower], ident_str, line, col, tok_len))
            # Check coordinate variables
            elif ident_lower in _VARIABLES:
                tokens.append(Token(TokenType.VARIABLE, ident_lower, ident_str, line, col, tok_len))
            # Check continuous parameters
            else:
                param_match = _PARAM_PATTERN.match(ident_lower)
                if param_match:
                    prefix, index_str = param_match.groups()
                    param_index = int(index_str)
                    tokens.append(Token(TokenType.PARAMETER, (ident_lower, param_index), ident_str, line, col, tok_len))
                else:
                    # Generic identifier (function name)
                    tokens.append(Token(TokenType.IDENTIFIER, ident_lower, ident_str, line, col, tok_len))

            idx += tok_len
            col += tok_len
            continue

        # If we reached here, unrecognized character
        raise LexerError(f"Invalid character {ch!r}", line=line, column=col, source_text=source_text)

    # Append EOF token
    tokens.append(Token(TokenType.EOF, None, "", line, col, 0))

    if not insert_implicit_mul:
        return tokens

    # Insert implicit multiplication
    return _insert_implicit_multiplication(tokens, source_text)


def _can_end_atom(tok: Token) -> bool:
    return tok.type in (
        TokenType.NUMBER,
        TokenType.CONSTANT,
        TokenType.VARIABLE,
        TokenType.PARAMETER,
        TokenType.RPAREN,
    )


def _can_start_atom(tok: Token) -> bool:
    return tok.type in (
        TokenType.NUMBER,
        TokenType.CONSTANT,
        TokenType.VARIABLE,
        TokenType.PARAMETER,
        TokenType.IDENTIFIER,
        TokenType.LPAREN,
    )


def _insert_implicit_multiplication(tokens: List[Token], source_text: str) -> List[Token]:
    """
    Inserts implicit STAR tokens between adjacent operands where mathematical
    juxtaposition is implied (e.g. 2x -> 2 * x, (x+1)(y+1) -> (x+1) * (y+1)).
    """
    if len(tokens) <= 1:
        return tokens

    result: List[Token] = []
    for i in range(len(tokens) - 1):
        cur = tokens[i]
        nxt = tokens[i + 1]
        result.append(cur)

        if _can_end_atom(cur) and _can_start_atom(nxt):
            star_col = cur.column + cur.length
            implicit_star = Token(TokenType.STAR, "*", "*", line=cur.line, column=star_col, length=0)
            result.append(implicit_star)

    result.append(tokens[-1])  # Append EOF
    return result
