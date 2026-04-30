"""
Genetic Programming engine – corrected with feature standardisation and robust fitness.
"""

import random
import numpy as np
from deap import base, creator, gp, tools, algorithms
import operator
import math
import functools
import config

# ========== Protected operators ==========
def protected_div(x, y):
    return x / y if abs(y) > 1e-6 else 1.0

def protected_sqrt(x):
    return math.sqrt(abs(x))

def protected_log(x):
    x = max(min(x, 1e6), 1e-6)
    return math.log(abs(x) + 1e-6)

def protected_exp(x):
    x = max(min(x, 100), -100)
    return math.exp(x)

def if_then_else(cond, true_val, false_val):
    return true_val if cond > 0 else false_val

def sigmoid(x):
    x = max(min(x, 100), -100)
    return 1.0 / (1.0 + math.exp(-x))

def clip_signal(x):
    return max(0.0, min(1.0, x))

# ========== Primitive set ==========
def build_primitive_set(feature_names):
    pset = gp.PrimitiveSet("main", len(feature_names))
    pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(feature_names)})
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(operator.neg, 1)
    pset.addPrimitive(protected_sqrt, 1)
    pset.addPrimitive(protected_log, 1)
    pset.addPrimitive(protected_exp, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(sigmoid, 1)
    pset.addPrimitive(if_then_else, 3)
    pset.addEphemeralConstant("rand", functools.partial(random.uniform, -1, 1))
    pset.addTerminal(0.0)
    pset.addTerminal(0.5)
    pset.addTerminal(1.0)
    return pset

def evaluate_individual(individual, pset, train_test_pairs, trans_cost):
    """
    Evaluate Sortino ratio across walk‑forward folds.
    train_test_pairs: list of (X_train, y_train, X_test, y_test, rf_test)
    """
    func = gp.compile(individual, pset)
    total_sortino = 0.0
    n_folds = len(train_test_pairs)
    for X_train, y_train, X_test, y_test, rf_test in train_test_pairs:
        # Generate signals on test set
        signals = []
        for row in X_test:
            try:
                val = func(*row)
                signals.append(clip_signal(val))
            except:
                signals.append(0.5)
        signals = np.array(signals)
        # Simple returns from log returns
        simple_returns = np.exp(y_test) - 1.0
        daily_pnl = signals * simple_returns
        # Transaction costs
        signal_changes = np.abs(np.diff(np.concatenate(([0], signals))))
        transaction_costs = signal_changes * trans_cost
        net_pnl = daily_pnl - transaction_costs
        excess = net_pnl - rf_test
        downside = excess[excess < 0].std()
        if downside == 0:
            sortino = 0.0
        else:
            sortino = (excess.mean() / downside) * np.sqrt(252)
        total_sortino += sortino
    avg_sortino = total_sortino / n_folds if n_folds > 0 else -1e6
    penalty = config.PARSIMONY_COEFF * (individual.height)
    return avg_sortino - penalty,

def run_gp_for_window(ticker, train_test_pairs, feature_names):
    """
    Evolve a formula for a single ticker on a fixed historical window.
    train_test_pairs: list of (X_train, y_train, X_test, y_test, rf_test)
    """
    pset = build_primitive_set(feature_names)
    # Reset DEAP creator
    try:
        del creator.FitnessMax
        del creator.Individual
    except:
        pass
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=config.INIT_DEPTH_MIN, max_=config.INIT_DEPTH_MAX)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    def eval_func(individual):
        return evaluate_individual(individual, pset, train_test_pairs, config.TRANSACTION_COST)
    toolbox.register("evaluate", eval_func)
    toolbox.register("select", tools.selTournament, tournsize=config.TOURNAMENT_SIZE)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, pset=pset, min_=0, max_=config.MAX_DEPTH)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.MAX_DEPTH))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.MAX_DEPTH))
    
    pop = toolbox.population(n=config.POPULATION_SIZE)
    hof = tools.HallOfFame(config.HALL_OF_FAME_SIZE)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    algorithms.eaSimple(pop, toolbox, cxpb=config.CROSSOVER_PROB, mutpb=config.MUTATION_PROB,
                        ngen=config.GENERATIONS, stats=stats, halloffame=hof, verbose=False)
    best = hof[0]
    return best, str(best)
