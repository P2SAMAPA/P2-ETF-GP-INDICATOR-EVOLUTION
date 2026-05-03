"""gp_evolver.py — Genetic Programming engine for indicator evolution."""

from __future__ import annotations

import functools
import operator
import random

import numpy as np
from deap import algorithms, base, creator, gp, tools

import config


# ── Primitives ────────────────────────────────────────────────────────────────
def protected_div(x: float, y: float) -> float:
    return x / y if abs(y) > 1e-6 else 1.0


def protected_sqrt(x: float) -> float:
    return np.sqrt(abs(x))


def protected_log(x: float) -> float:
    return np.log(abs(x)) if abs(x) > 1e-6 else 0.0


def sig(x: float) -> float:
    """Sigmoid — squashes output to (0, 1) for position sizing."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def build_pset(feature_names: list[str]) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("main", len(feature_names))
    pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(feature_names)})

    # Arithmetic
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)

    # Unary — richer feature transformations
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(protected_sqrt, 1)
    pset.addPrimitive(sig, 1)

    # Constants
    pset.addEphemeralConstant("rand", functools.partial(random.uniform, -2, 2))
    pset.addTerminal(0.0)
    pset.addTerminal(0.5)
    pset.addTerminal(1.0)
    pset.addTerminal(-1.0)

    return pset


# ── Fitness ───────────────────────────────────────────────────────────────────
def _evaluate(
    individual: gp.PrimitiveTree,
    pset: gp.PrimitiveSet,
    X_test: np.ndarray,
    y_test: np.ndarray,
    rf_test: np.ndarray,
) -> tuple[float]:
    """Sortino-ratio fitness (penalises downside volatility only)."""
    func = gp.compile(individual, pset)
    signals = np.empty(len(X_test))
    for i, row in enumerate(X_test):
        try:
            v = func(*row)
            signals[i] = float(np.clip(v if np.isfinite(v) else 0.5, 0.0, 1.0))
        except Exception:
            signals[i] = 0.5

    returns = np.expm1(y_test)  # convert log-returns to simple returns
    pnl = signals * returns

    # Transaction costs
    changes = np.abs(np.diff(np.concatenate(([0.0], signals))))
    pnl -= changes * config.TRANSACTION_COST

    excess = pnl - rf_test
    downside = excess[excess < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-6
    if downside_std < 1e-8:
        sortino = 0.0
    else:
        sortino = (excess.mean() / downside_std) * np.sqrt(252)

    # Parsimony pressure — penalise overly complex trees
    complexity_penalty = 0.001 * len(individual)
    return (sortino - complexity_penalty,)


def run_gp(
    ticker: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    rf_test: np.ndarray,
    feature_names: list[str],
    seed: int = 42,
) -> tuple[float, str]:
    """Run GP evolution and return (best_sortino_score, formula_string)."""
    random.seed(seed)
    np.random.seed(seed)

    pset = build_pset(feature_names)

    # Clean up any leftover DEAP creator state
    for name in ("FitnessMax", "Individual"):
        if name in creator.__dict__:
            delattr(creator, name)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register(
        "expr",
        gp.genHalfAndHalf,
        pset=pset,
        min_=config.INIT_DEPTH_MIN,
        max_=config.INIT_DEPTH_MAX,
    )
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register(
        "evaluate",
        _evaluate,
        pset=pset,
        X_test=X_test,
        y_test=y_test,
        rf_test=rf_test,
    )
    toolbox.register("select", tools.selTournament, tournsize=config.TOURNAMENT_SIZE)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, pset=pset, min_=0, max_=config.MAX_DEPTH)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    depth_limit = gp.staticLimit(
        key=operator.attrgetter("height"), max_value=config.MAX_DEPTH
    )
    toolbox.decorate("mate", depth_limit)
    toolbox.decorate("mutate", depth_limit)

    pop = toolbox.population(n=config.POPULATION_SIZE)
    hof = tools.HallOfFame(config.HALL_OF_FAME_SIZE)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=config.CROSSOVER_PROB,
        mutpb=config.MUTATION_PROB,
        ngen=config.GENERATIONS,
        halloffame=hof,
        stats=stats,
        verbose=False,
    )

    best = hof[0]
    best_score = float(best.fitness.values[0])
    best_formula = str(best)
    return best_score, best_formula
