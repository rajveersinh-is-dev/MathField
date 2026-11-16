"""
MathField Core Abstract Syntax Tree (AST) Package.
"""

from mathfield.core.ast.tokens import (
    LexerError,
    Token,
    TokenType,
    tokenize,
)
from mathfield.core.ast.nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    FUNCTION_ARITIES,
    FunctionCallNode,
    ParameterNode,
    UnaryOpNode,
    VariableNode,
    validate_parameter_bounds,
)
from mathfield.core.ast.parser import (
    ParseError,
    Parser,
    parse,
)
from mathfield.core.ast.transforms import (
    diff,
    is_constant,
    is_neg_one,
    is_one,
    is_zero,
    simplify,
)
from mathfield.core.ast.domain import (
    PoleWarning,
    clamp_fn,
    protected_acos,
    protected_asin,
    protected_cbrt,
    protected_exp,
    protected_log,
    protected_pow,
    protected_sinc,
    protected_sqrt,
    protected_tan,
    rbf_gaussian,
    rbf_multiquadric,
    rbf_wendland,
    regularized_div,
    sigmoid_fn,
    softplus_fn,
)

__all__ = [
    # Tokens
    "Token",
    "TokenType",
    "LexerError",
    "tokenize",
    # Nodes
    "ASTNode",
    "ConstantNode",
    "VariableNode",
    "ParameterNode",
    "UnaryOpNode",
    "BinaryOpNode",
    "FunctionCallNode",
    "FUNCTION_ARITIES",
    "validate_parameter_bounds",
    # Parser
    "Parser",
    "ParseError",
    "parse",
    # Transforms
    "simplify",
    "diff",
    "is_constant",
    "is_zero",
    "is_one",
    "is_neg_one",
    # Domain
    "PoleWarning",
    "regularized_div",
    "protected_log",
    "protected_exp",
    "protected_sqrt",
    "protected_cbrt",
    "protected_pow",
    "protected_sinc",
    "protected_tan",
    "protected_asin",
    "protected_acos",
    "rbf_gaussian",
    "rbf_multiquadric",
    "rbf_wendland",
    "sigmoid_fn",
    "softplus_fn",
    "clamp_fn",
]
