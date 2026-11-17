"""
mathfield/core/optimize/evolutionary.py

Stage 4: Deterministic Strongly-Typed Genetic Programming (STGP).
Bit-exact reproducible symbolic mathematical discovery minimizing:
J(M) = E(M) + lambda * C(M)
Zero lookup tables or direct raster memorization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from mathfield.core.ast.nodes import (
    ASTNode,
    BinaryOpNode,
    ConstantNode,
    FunctionCallNode,
    UnaryOpNode,
    VariableNode,
)
from mathfield.core.engine.cpu_numpy import evaluate_numpy

_UNARY_FUNCS = ["sin", "cos", "exp", "log", "sqrt", "abs", "tanh", "sigmoid"]
_BINARY_OPS = ["+", "-", "*", "/"]


@dataclass
class Individual:
    ast: ASTNode
    mse: float = float("inf")
    complexity: int = 0
    fitness: float = float("inf")


class EvolutionaryOptimizer:
    """
    Deterministic Genetic Programming Engine for continuous mathematical discovery.
    """

    def __init__(
        self,
        random_seed: int = 42,
        population_size: int = 40,
        generations: int = 15,
        lambda_reg: float = 0.01,
        tournament_size: int = 3,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.25,
        max_depth: int = 5,
    ) -> None:
        self.random_seed = random_seed
        self.population_size = population_size
        self.generations = generations
        self.lambda_reg = lambda_reg
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_depth = max_depth

        self.rng = np.random.Generator(np.random.PCG64(random_seed))
        self.pareto_front: List[Individual] = []

    def _random_terminal(self) -> ASTNode:
        r = self.rng.random()
        if r < 0.45:
            return VariableNode("x")
        elif r < 0.90:
            return VariableNode("y")
        else:
            val = float(np.round(self.rng.uniform(-2.0, 2.0), 3))
            return ConstantNode(val)

    def _random_tree(self, depth: int, method: str = "grow") -> ASTNode:
        if depth <= 1 or (method == "grow" and self.rng.random() < 0.35):
            return self._random_terminal()

        # Generate internal node
        if self.rng.random() < 0.5:
            # Unary function
            fn = str(self.rng.choice(_UNARY_FUNCS))
            child = self._random_tree(depth - 1, method)
            return FunctionCallNode(fn, (child,))
        else:
            # Binary operator
            op = str(self.rng.choice(_BINARY_OPS))
            left = self._random_tree(depth - 1, method)
            right = self._random_tree(depth - 1, method)
            return BinaryOpNode(op, left, right)

    def _generate_population(self) -> List[Individual]:
        pop: List[Individual] = []
        for i in range(self.population_size):
            depth = int(self.rng.integers(2, self.max_depth + 1))
            method = "full" if (i % 2 == 0) else "grow"
            tree = self._random_tree(depth, method)
            pop.append(Individual(ast=tree))
        return pop

    def _evaluate_individual(
        self,
        ind: Individual,
        X: np.ndarray,
        Y: np.ndarray,
        target: np.ndarray,
    ) -> None:
        try:
            pred = evaluate_numpy(ind.ast, X, Y, np.array([]))
            diff = pred - target
            # Guard against NaN/Inf
            if not np.all(np.isfinite(diff)):
                ind.mse = 1e6
            else:
                ind.mse = float(np.mean(diff**2))
        except Exception:
            ind.mse = 1e6

        ind.complexity = ind.ast.complexity()
        ind.fitness = ind.mse + self.lambda_reg * ind.complexity

    def _tournament_select(self, pop: List[Individual]) -> Individual:
        indices = self.rng.choice(len(pop), size=self.tournament_size, replace=False)
        best = pop[indices[0]]
        for idx in indices[1:]:
            cand = pop[idx]
            if (
                cand.fitness < best.fitness
                or abs(cand.fitness - best.fitness) < 1e-9
                and cand.complexity < best.complexity
            ):
                best = cand
        return best

    def _collect_subtrees(self, node: ASTNode) -> List[ASTNode]:
        nodes = [node]
        if isinstance(node, BinaryOpNode):
            nodes.extend(self._collect_subtrees(node.left))
            nodes.extend(self._collect_subtrees(node.right))
        elif isinstance(node, FunctionCallNode):
            for arg in node.args:
                nodes.extend(self._collect_subtrees(arg))
        elif isinstance(node, UnaryOpNode):
            nodes.extend(self._collect_subtrees(node.operand))
        return nodes

    def _replace_subtree(self, root: ASTNode, target: ASTNode, replacement: ASTNode) -> ASTNode:
        if root is target:
            return replacement
        if isinstance(root, BinaryOpNode):
            new_left = self._replace_subtree(root.left, target, replacement)
            new_right = self._replace_subtree(root.right, target, replacement)
            return BinaryOpNode(root.op, new_left, new_right)
        elif isinstance(root, FunctionCallNode):
            new_args = tuple(self._replace_subtree(arg, target, replacement) for arg in root.args)
            return FunctionCallNode(root.func, new_args)
        elif isinstance(root, UnaryOpNode):
            new_op = self._replace_subtree(root.operand, target, replacement)
            return UnaryOpNode(root.op, new_op)
        return root

    def _crossover(self, parent1: ASTNode, parent2: ASTNode) -> Tuple[ASTNode, ASTNode]:
        nodes1 = self._collect_subtrees(parent1)
        nodes2 = self._collect_subtrees(parent2)

        target1 = nodes1[int(self.rng.integers(0, len(nodes1)))]
        target2 = nodes2[int(self.rng.integers(0, len(nodes2)))]

        child1 = self._replace_subtree(parent1, target1, target2)
        child2 = self._replace_subtree(parent2, target2, target1)
        return child1, child2

    def _mutate(self, tree: ASTNode) -> ASTNode:
        nodes = self._collect_subtrees(tree)
        target = nodes[int(self.rng.integers(0, len(nodes)))]

        r = self.rng.random()
        if r < 0.5:
            # Subtree replacement
            replacement = self._random_tree(depth=2, method="grow")
        elif isinstance(target, ConstantNode):
            # Perturb constant
            replacement = ConstantNode(float(target.value + self.rng.normal(0, 0.2)))
        elif isinstance(target, VariableNode):
            # Swap x <-> y
            new_var = "y" if target.name == "x" else "x"
            replacement = VariableNode(new_var)
        elif isinstance(target, FunctionCallNode):
            new_fn = str(self.rng.choice(_UNARY_FUNCS))
            replacement = FunctionCallNode(new_fn, target.args)
        elif isinstance(target, BinaryOpNode):
            new_op = str(self.rng.choice(_BINARY_OPS))
            replacement = BinaryOpNode(new_op, target.left, target.right)
        else:
            replacement = self._random_terminal()

        return self._replace_subtree(tree, target, replacement)

    def _update_pareto(self, individual: Individual) -> None:
        """Maintain non-dominated archive in (mse, complexity)."""
        is_dominated = False
        new_front = []
        for p in self.pareto_front:
            if p.mse <= individual.mse and p.complexity <= individual.complexity:
                if p.mse < individual.mse or p.complexity < individual.complexity:
                    is_dominated = True
                    new_front.append(p)
                    break
            elif individual.mse <= p.mse and individual.complexity <= p.complexity:
                # p is dominated by individual, do not keep p
                continue
            else:
                new_front.append(p)

        if not is_dominated:
            new_front.append(individual)
            self.pareto_front = new_front

    def evolve(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        target: np.ndarray,
        generations: Optional[int] = None,
        callback: Optional[Callable[[int, Individual], None]] = None,
    ) -> Tuple[ASTNode, Dict[str, Any]]:
        """
        Execute deterministic evolutionary search loop.
        """
        n_gens = generations if generations is not None else self.generations
        population = self._generate_population()

        for ind in population:
            self._evaluate_individual(ind, X, Y, target)
            self._update_pareto(ind)

        best_overall = min(population, key=lambda ind: ind.fitness)

        for gen in range(n_gens):
            # Elitism: retain top 2
            sorted_pop = sorted(population, key=lambda ind: ind.fitness)
            next_pop = [sorted_pop[0], sorted_pop[1]] if len(sorted_pop) >= 2 else list(sorted_pop)

            while len(next_pop) < self.population_size:
                p1 = self._tournament_select(population)
                p2 = self._tournament_select(population)

                if self.rng.random() < self.crossover_prob:
                    c1_ast, c2_ast = self._crossover(p1.ast, p2.ast)
                else:
                    c1_ast, c2_ast = p1.ast, p2.ast

                if self.rng.random() < self.mutation_prob:
                    c1_ast = self._mutate(c1_ast)
                if self.rng.random() < self.mutation_prob:
                    c2_ast = self._mutate(c2_ast)

                ind1 = Individual(ast=c1_ast)
                ind2 = Individual(ast=c2_ast)

                self._evaluate_individual(ind1, X, Y, target)
                self._evaluate_individual(ind2, X, Y, target)

                self._update_pareto(ind1)
                self._update_pareto(ind2)

                next_pop.append(ind1)
                if len(next_pop) < self.population_size:
                    next_pop.append(ind2)

            population = next_pop
            gen_best = min(population, key=lambda ind: ind.fitness)
            if gen_best.fitness < best_overall.fitness:
                best_overall = gen_best

            if callback is not None:
                callback(gen, best_overall)

        diagnostics = {
            "best_mse": best_overall.mse,
            "best_complexity": best_overall.complexity,
            "best_fitness": best_overall.fitness,
            "pareto_size": len(self.pareto_front),
        }
        return best_overall.ast, diagnostics
