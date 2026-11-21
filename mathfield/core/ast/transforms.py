"""
Algebraic simplification and analytical symbolic differentiation for MathField AST.
"""

from __future__ import annotations

import math
from typing import Optional

from mathfield.core.ast.nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    FunctionCallNode,
    ParameterNode,
    UnaryOpNode,
    VariableNode,
)


def is_constant(node: ASTNode, val: Optional[float] = None) -> bool:
    """Check if node is a ConstantNode, optionally matching a specific numerical value."""
    if not isinstance(node, ConstantNode):
        return False
    if val is None:
        return True
    return math.isclose(node.value, val, abs_tol=1e-12)


def is_zero(node: ASTNode) -> bool:
    return is_constant(node, 0.0)


def is_one(node: ASTNode) -> bool:
    return is_constant(node, 1.0)


def is_neg_one(node: ASTNode) -> bool:
    return is_constant(node, -1.0)


def _is_integer_value(value: float) -> bool:
    return math.isfinite(value) and math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-12)


def _is_nonnegative_node(node: ASTNode) -> bool:
    """Conservative check for expressions known to be non-negative for all real inputs."""
    if isinstance(node, ConstantNode):
        return node.value >= 0.0
    if isinstance(node, FunctionCallNode):
        return node.func in ("abs", "square", "exp", "sqrt", "softplus", "sigmoid")
    if isinstance(node, BinaryOpNode) and node.op in ("^", "**"):
        return isinstance(node.right, ConstantNode) and _is_integer_value(node.right.value) and round(node.right.value) % 2 == 0
    return False


def simplify(node: ASTNode, max_iterations: int = 50) -> ASTNode:
    """
    Applies fixed-point bottom-up algebraic simplification rules to the AST.
    Iterates until convergence or max_iterations is reached.
    """
    current = node
    for _ in range(max_iterations):
        reduced = _simplify_step(current)
        if reduced == current:
            break
        current = reduced
    return current


def _simplify_step(node: ASTNode) -> ASTNode:
    """Single post-order recursive simplification traversal."""
    if isinstance(node, (ConstantNode, VariableNode, ParameterNode)):
        return node

    elif isinstance(node, UnaryOpNode):
        operand = _simplify_step(node.operand)
        op = node.op

        if op == "+":
            return operand

        if op == "-":
            # Constant folding: -C -> -C
            if isinstance(operand, ConstantNode):
                return ConstantNode(-operand.value)
            # Double negation: -(-u) -> u
            if isinstance(operand, UnaryOpNode) and operand.op == "-":
                return operand.operand
            # Negation of subtraction: -(a - b) -> b - a
            if isinstance(operand, BinaryOpNode) and operand.op == "-":
                return _simplify_step(BinaryOpNode("-", operand.right, operand.left))
            # Negation of zero: -0 -> 0
            if is_zero(operand):
                return ConstantNode(0.0)

            return UnaryOpNode("-", operand)

    elif isinstance(node, BinaryOpNode):
        left = _simplify_step(node.left)
        right = _simplify_step(node.right)
        op = node.op

        # 1. Constant Folding
        if isinstance(left, ConstantNode) and isinstance(right, ConstantNode):
            lv, rv = left.value, right.value
            if op == "+":
                return ConstantNode(lv + rv)
            elif op == "-":
                return ConstantNode(lv - rv)
            elif op == "*":
                return ConstantNode(lv * rv)
            elif op == "/":
                if abs(rv) > 1e-15:
                    return ConstantNode(lv / rv)
            elif op in ("^", "**"):
                try:
                    if abs(lv) < 1e-15 and abs(rv) < 1e-15:
                        return ConstantNode(1.0)
                    if lv >= 0 or rv.is_integer():
                        res = math.pow(lv, rv)
                        if not math.isnan(res) and not math.isinf(res):
                            return ConstantNode(res)
                except (OverflowError, ValueError):
                    pass
            elif op == "%" and abs(rv) > 1e-15:
                return ConstantNode(math.fmod(lv, rv))

        # 2. Additive Identities
        if op == "+":
            if is_zero(left):
                return right
            if is_zero(right):
                return left
            # u + (-w) -> u - w
            if isinstance(right, UnaryOpNode) and right.op == "-":
                return _simplify_step(BinaryOpNode("-", left, right.operand))
            # (-u) + w -> w - u
            if isinstance(left, UnaryOpNode) and left.op == "-":
                return _simplify_step(BinaryOpNode("-", right, left.operand))
            # Reassociation: (u + C1) + C2 -> u + (C1 + C2)
            if (
                isinstance(left, BinaryOpNode)
                and left.op == "+"
                and isinstance(left.right, ConstantNode)
                and isinstance(right, ConstantNode)
            ):
                return _simplify_step(BinaryOpNode("+", left.left, ConstantNode(left.right.value + right.value)))

        elif op == "-":
            if is_zero(right):
                return left
            if is_zero(left):
                return _simplify_step(UnaryOpNode("-", right))
            # Structural identity: u - u -> 0
            if left == right:
                return ConstantNode(0.0)
            # u - (-w) -> u + w
            if isinstance(right, UnaryOpNode) and right.op == "-":
                return _simplify_step(BinaryOpNode("+", left, right.operand))

        # 3. Multiplicative Identities
        elif op == "*":
            if is_zero(left) or is_zero(right):
                return ConstantNode(0.0)
            if is_one(left):
                return right
            if is_one(right):
                return left
            if is_neg_one(left):
                return _simplify_step(UnaryOpNode("-", right))
            if is_neg_one(right):
                return _simplify_step(UnaryOpNode("-", left))
            # (-u) * (-w) -> u * w
            if isinstance(left, UnaryOpNode) and left.op == "-" and isinstance(right, UnaryOpNode) and right.op == "-":
                return _simplify_step(BinaryOpNode("*", left.operand, right.operand))
            # (-u) * w -> -(u * w)
            if isinstance(left, UnaryOpNode) and left.op == "-":
                return _simplify_step(UnaryOpNode("-", BinaryOpNode("*", left.operand, right)))
            # u * (-w) -> -(u * w)
            if isinstance(right, UnaryOpNode) and right.op == "-":
                return _simplify_step(UnaryOpNode("-", BinaryOpNode("*", left, right.operand)))
            # Reassociation: (u * C1) * C2 -> u * (C1 * C2)
            if (
                isinstance(left, BinaryOpNode)
                and left.op == "*"
                and isinstance(left.right, ConstantNode)
                and isinstance(right, ConstantNode)
            ):
                return _simplify_step(BinaryOpNode("*", left.left, ConstantNode(left.right.value * right.value)))

        # 4. Division Identities
        elif op == "/":
            if is_zero(left) and not is_zero(right):
                return ConstantNode(0.0)
            if is_one(right):
                return left
            if left == right and not is_zero(right):
                return ConstantNode(1.0)
            # (-u) / (-w) -> u / w
            if isinstance(left, UnaryOpNode) and left.op == "-" and isinstance(right, UnaryOpNode) and right.op == "-":
                return _simplify_step(BinaryOpNode("/", left.operand, right.operand))
            # (-u) / w -> -(u / w)
            if isinstance(left, UnaryOpNode) and left.op == "-":
                return _simplify_step(UnaryOpNode("-", BinaryOpNode("/", left.operand, right)))
            # u / (-w) -> -(u / w)
            if isinstance(right, UnaryOpNode) and right.op == "-":
                return _simplify_step(UnaryOpNode("-", BinaryOpNode("/", left, right.operand)))

        # 5. Power Identities
        elif op in ("^", "**"):
            if is_zero(right):
                return ConstantNode(1.0)
            if is_one(right):
                return left
            if is_one(left):
                return ConstantNode(1.0)
            if is_zero(left) and isinstance(right, ConstantNode) and right.value > 0:
                return ConstantNode(0.0)
            # (u ^ a) ^ b -> u ^ (a * b), but only when the reassociation is
            # valid for real-valued, possibly negative u. With integer inner
            # exponents the parity convention of protected_pow is preserved;
            # otherwise only a provably non-negative base may be reassociated
            # (e.g. (x^2)^0.5 is |x|, *not* x, for negative x).
            if (
                isinstance(left, BinaryOpNode)
                and left.op in ("^", "**")
                and isinstance(left.right, ConstantNode)
                and isinstance(right, ConstantNode)
                and (
                    (_is_integer_value(left.right.value) and _is_integer_value(right.value))
                    or _is_nonnegative_node(left.left)
                )
            ):
                return _simplify_step(BinaryOpNode("^", left.left, ConstantNode(left.right.value * right.value)))

        return BinaryOpNode(op, left, right)

    elif isinstance(node, FunctionCallNode):
        args = tuple(_simplify_step(arg) for arg in node.args)
        f = node.func

        # Constant folding for functions
        if all(isinstance(a, ConstantNode) for a in args):
            try:
                val = node.eval_scalar(0.0, 0.0, [])
                if not math.isnan(val) and not math.isinf(val):
                    return ConstantNode(val)
            except Exception:
                pass

        # Specific algebraic reductions
        if f == "exp" and is_zero(args[0]):
            return ConstantNode(1.0)
        if f == "log" and is_one(args[0]):
            return ConstantNode(0.0)
        if f == "sin" and is_zero(args[0]):
            return ConstantNode(0.0)
        if f == "cos" and is_zero(args[0]):
            return ConstantNode(1.0)
        if f == "tan" and is_zero(args[0]):
            return ConstantNode(0.0)
        if f == "sinh" and is_zero(args[0]):
            return ConstantNode(0.0)
        if f == "cosh" and is_zero(args[0]):
            return ConstantNode(1.0)
        if f == "tanh" and is_zero(args[0]):
            return ConstantNode(0.0)
        if f == "sqrt" and is_zero(args[0]):
            return ConstantNode(0.0)
        if f == "sqrt" and is_one(args[0]):
            return ConstantNode(1.0)
        if f == "square" and is_zero(args[0]):
            return ConstantNode(0.0)
        if f == "square" and is_one(args[0]):
            return ConstantNode(1.0)
        if f == "pow":
            return _simplify_step(BinaryOpNode("^", args[0], args[1]))

        return FunctionCallNode(f, args)

    return node


def diff(node: ASTNode, target_var: str) -> ASTNode:
    """
    Computes the exact analytical symbolic derivative of ASTNode with respect to target_var.
    Includes intermediate simplification to prevent exponential tree blowup.

    Parameters:
        node: The expression ASTNode to differentiate.
        target_var: The variable or parameter name (e.g. 'x', 'y', 'c0', 'theta1').

    Returns:
        The simplified analytical derivative ASTNode.
    """
    norm_target = target_var.lower()
    raw_derivative = _diff_rec(node, norm_target)
    return simplify(raw_derivative)


def _diff_rec(node: ASTNode, target: str) -> ASTNode:
    """Recursive differentiation worker."""
    if isinstance(node, ConstantNode):
        return ConstantNode(0.0)

    elif isinstance(node, (VariableNode, ParameterNode)):
        return ConstantNode(1.0 if node.name == target else 0.0)

    elif isinstance(node, UnaryOpNode):
        du = _diff_rec(node.operand, target)
        if node.op == "+":
            return du
        elif node.op == "-":
            return simplify(UnaryOpNode("-", du))

    elif isinstance(node, BinaryOpNode):
        u = node.left
        w = node.right
        du = _diff_rec(u, target)
        dw = _diff_rec(w, target)
        op = node.op

        if op == "+":
            return simplify(BinaryOpNode("+", du, dw))
        elif op == "-":
            return simplify(BinaryOpNode("-", du, dw))
        elif op == "*":
            # Product Rule: u' * w + u * w'
            term1 = simplify(BinaryOpNode("*", du, w))
            term2 = simplify(BinaryOpNode("*", u, dw))
            return simplify(BinaryOpNode("+", term1, term2))
        elif op == "/":
            # Quotient Rule: (u' * w - u * w') / (w ^ 2)
            num1 = simplify(BinaryOpNode("*", du, w))
            num2 = simplify(BinaryOpNode("*", u, dw))
            num = simplify(BinaryOpNode("-", num1, num2))
            den = simplify(BinaryOpNode("^", w, ConstantNode(2.0)))
            return simplify(BinaryOpNode("/", num, den))
        elif op in ("^", "**"):
            # Power Rule:
            # If w doesn't depend on target: w * (u ^ (w - 1)) * u'
            if is_zero(dw):
                if is_zero(du):
                    return ConstantNode(0.0)
                w_minus_1 = simplify(BinaryOpNode("-", w, ConstantNode(1.0)))
                u_pow = simplify(BinaryOpNode("^", u, w_minus_1))
                return simplify(BinaryOpNode("*", simplify(BinaryOpNode("*", w, u_pow)), du))
            # If u doesn't depend on target: ln(u) * (u ^ w) * w'
            elif is_zero(du):
                ln_u = FunctionCallNode("log", (u,))
                u_pow = simplify(BinaryOpNode("^", u, w))
                return simplify(BinaryOpNode("*", simplify(BinaryOpNode("*", ln_u, u_pow)), dw))
            else:
                # General: u^w * (w' * ln(u) + w * (u' / u))
                u_pow = simplify(BinaryOpNode("^", u, w))
                ln_u = FunctionCallNode("log", (u,))
                term1 = simplify(BinaryOpNode("*", dw, ln_u))
                term2 = simplify(BinaryOpNode("*", w, simplify(BinaryOpNode("/", du, u))))
                bracket = simplify(BinaryOpNode("+", term1, term2))
                return simplify(BinaryOpNode("*", u_pow, bracket))
        elif op == "%":
            return du

    elif isinstance(node, FunctionCallNode):
        f = node.func
        args = node.args
        dargs = [_diff_rec(arg, target) for arg in args]

        # If no arguments depend on the target, derivative is 0
        if all(is_zero(da) for da in dargs):
            return ConstantNode(0.0)

        # Single-argument chain rule: f'(u) * u'
        u = args[0]
        du = dargs[0]

        if f == "sin":
            # sin'(u) = cos(u)
            return simplify(BinaryOpNode("*", FunctionCallNode("cos", (u,)), du))
        elif f == "cos":
            # cos'(u) = -sin(u)
            return simplify(BinaryOpNode("*", UnaryOpNode("-", FunctionCallNode("sin", (u,))), du))
        elif f == "tan":
            # tan'(u) = 1 + tan(u)^2
            sec2 = simplify(BinaryOpNode("+", ConstantNode(1.0), FunctionCallNode("square", (node,))))
            return simplify(BinaryOpNode("*", sec2, du))
        elif f == "sinc":
            # d/du [sin(pi * u) / (pi * u)] = (pi * u * cos(pi * u) - sin(pi * u)) / (pi * u^2)
            # If du is 0, already returned 0 above.
            pi_u = BinaryOpNode("*", ConstantNode(math.pi, symbol="pi"), u)
            num = BinaryOpNode(
                "-", BinaryOpNode("*", pi_u, FunctionCallNode("cos", (pi_u,))), FunctionCallNode("sin", (pi_u,))
            )
            den = BinaryOpNode("*", ConstantNode(math.pi, symbol="pi"), BinaryOpNode("^", u, ConstantNode(2.0)))
            return simplify(BinaryOpNode("*", BinaryOpNode("/", num, den), du))
        elif f == "asin":
            # asin'(u) = 1 / sqrt(1 - u^2)
            den = FunctionCallNode(
                "sqrt", (BinaryOpNode("-", ConstantNode(1.0), BinaryOpNode("^", u, ConstantNode(2.0))),)
            )
            return simplify(BinaryOpNode("/", du, den))
        elif f == "acos":
            # acos'(u) = -1 / sqrt(1 - u^2)
            den = FunctionCallNode(
                "sqrt", (BinaryOpNode("-", ConstantNode(1.0), BinaryOpNode("^", u, ConstantNode(2.0))),)
            )
            return simplify(BinaryOpNode("/", UnaryOpNode("-", du), den))
        elif f == "atan":
            # atan'(u) = 1 / (1 + u^2)
            den = BinaryOpNode("+", ConstantNode(1.0), BinaryOpNode("^", u, ConstantNode(2.0)))
            return simplify(BinaryOpNode("/", du, den))
        elif f == "sinh":
            return simplify(BinaryOpNode("*", FunctionCallNode("cosh", (u,)), du))
        elif f == "cosh":
            return simplify(BinaryOpNode("*", FunctionCallNode("sinh", (u,)), du))
        elif f == "tanh":
            # tanh'(u) = 1 - tanh(u)^2
            sech2 = BinaryOpNode("-", ConstantNode(1.0), BinaryOpNode("^", node, ConstantNode(2.0)))
            return simplify(BinaryOpNode("*", sech2, du))
        elif f == "exp":
            return simplify(BinaryOpNode("*", node, du))
        elif f == "log":
            return simplify(BinaryOpNode("/", du, u))
        elif f == "sqrt":
            # sqrt'(u) = 1 / (2 * sqrt(u))
            den = BinaryOpNode("*", ConstantNode(2.0), node)
            return simplify(BinaryOpNode("/", du, den))
        elif f == "cbrt":
            # cbrt'(u) = 1 / (3 * cbrt(u)^2)
            cbrt2 = BinaryOpNode("^", node, ConstantNode(2.0))
            den = BinaryOpNode("*", ConstantNode(3.0), cbrt2)
            return simplify(BinaryOpNode("/", du, den))
        elif f == "square":
            # square'(u) = 2 * u * u'
            return simplify(BinaryOpNode("*", BinaryOpNode("*", ConstantNode(2.0), u), du))
        elif f == "abs":
            # abs'(u) = sign(u) * u'
            return simplify(BinaryOpNode("*", FunctionCallNode("sign", (u,)), du))
        elif f == "sign":
            return ConstantNode(0.0)
        elif f == "sigmoid":
            # sigmoid'(u) = sigmoid(u) * (1 - sigmoid(u)) * u'
            one_minus_sig = BinaryOpNode("-", ConstantNode(1.0), node)
            return simplify(BinaryOpNode("*", BinaryOpNode("*", node, one_minus_sig), du))
        elif f == "softplus":
            # softplus'(u) = sigmoid(u) * u'
            return simplify(BinaryOpNode("*", FunctionCallNode("sigmoid", (u,)), du))
        elif f == "clamp":
            # Subgradient: derivative is u' inside the clamp interval
            return du
        elif f == "pow":
            return _diff_rec(BinaryOpNode("^", args[0], args[1]), target)
        elif f == "rbf_gaussian":
            # phi(r, eps) = exp(-(eps * r)^2)
            # d phi / d target = -2 * eps * (r * deps + eps * dr) * phi
            r, eps = args[0], args[1]
            dr, deps = dargs[0], dargs[1]
            eps_r = BinaryOpNode("*", eps, r)
            d_eps_r = BinaryOpNode("+", BinaryOpNode("*", deps, r), BinaryOpNode("*", eps, dr))
            d_arg = BinaryOpNode("*", BinaryOpNode("*", ConstantNode(-2.0), eps_r), d_eps_r)
            return simplify(BinaryOpNode("*", node, d_arg))
        elif f == "rbf_multiquadric":
            # phi(r, eps) = sqrt(1 + (eps * r)^2)
            # d phi = (eps * r * (r * deps + eps * dr)) / phi
            r, eps = args[0], args[1]
            dr, deps = dargs[0], dargs[1]
            eps_r = BinaryOpNode("*", eps, r)
            d_eps_r = BinaryOpNode("+", BinaryOpNode("*", deps, r), BinaryOpNode("*", eps, dr))
            num = BinaryOpNode("*", eps_r, d_eps_r)
            return simplify(BinaryOpNode("/", num, node))
        elif f == "rbf_wendland":
            # phi(r, R) = (1 - r/R)_+^4 * (4 * r/R + 1)
            # d phi / dr = -20 * (r / R^2) * (1 - r/R)^3
            # d phi / dR = 20 * (r^2 / R^3) * (1 - r/R)^3
            r, rad = args[0], args[1]
            dr, drad = dargs[0], dargs[1]
            # The kernel itself clamps r to non-negative values; the derivative
            # must use the same clamped radius and vanish outside the support.
            r = FunctionCallNode(
                "clamp", (r, ConstantNode(0.0), ConstantNode(1e12))
            )
            r_over_R = BinaryOpNode("/", r, rad)
            # (1 - r/R)^3 must be clamped at zero: outside the compact support
            # the kernel and therefore its derivative are exactly zero.
            t = FunctionCallNode(
                "clamp",
                (BinaryOpNode("-", ConstantNode(1.0), r_over_R), ConstantNode(0.0), ConstantNode(1.0)),
            )
            t3 = BinaryOpNode("^", t, ConstantNode(3.0))

            term_dr = ConstantNode(0.0)
            if not is_zero(dr):
                factor_r = BinaryOpNode(
                    "/", BinaryOpNode("*", ConstantNode(-20.0), r), BinaryOpNode("^", rad, ConstantNode(2.0))
                )
                term_dr = BinaryOpNode("*", BinaryOpNode("*", factor_r, t3), dr)

            term_drad = ConstantNode(0.0)
            if not is_zero(drad):
                factor_R = BinaryOpNode(
                    "/",
                    BinaryOpNode("*", ConstantNode(20.0), BinaryOpNode("^", r, ConstantNode(2.0))),
                    BinaryOpNode("^", rad, ConstantNode(3.0)),
                )
                term_drad = BinaryOpNode("*", BinaryOpNode("*", factor_R, t3), drad)

            return simplify(BinaryOpNode("+", term_dr, term_drad))

    raise ValueError(f"Unknown or unsupported AST node: {type(node)}")
