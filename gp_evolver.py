"""
Genetic Programming engine using DEAP.
Evolves trading indicators (long‑only) with Sortino fitness and transaction costs.
"""

import random
import numpy as np
import pandas as pd
from deap import base, creator, gp, tools, algorithms
import operator
import math

# Protected operators
def protected_div(x, y):
    return x / y if abs(y) > 1e-6 else 1.0

def protected_sqrt(x):
    return math.sqrt(abs(x))

def protected_log(x):
    return math.log(abs(x) + 1e-6)

def if_then_else(cond, true_val, false_val):
    return true_val if cond > 0 else false_val

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def clip_signal(x):
    return max(0.0, min(1.0, x))

# Primitive set
def build_primitive_set(feature_names):
    pset = gp.PrimitiveSet("main", len(feature_names))
    pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(feature_names)})
    
    # Arithmetic
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(operator.neg, 1)
    
    # Math
    pset.addPrimitive(protected_sqrt, 1)
    pset.addPrimitive(protected_log, 1)
    pset.addPrimitive(math.exp, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(sigmoid, 1)
    
    # Logical
    pset.addPrimitive(if_then_else, 3)
    
    # Terminal constants
    pset.addEphemeralConstant("rand", lambda: random.uniform(-1, 1))
    pset.addTerminal(0.0)
    pset.addTerminal(0.5)
    pset.addTerminal(1.0)
    
    return pset

def evaluate_individual(individual, pset, X_train, y_train, X_test, y_test, risk_free_rates, trans_cost):
    """
    Evaluate Sortino ratio on walk‑forward folds.
    individual: GP tree
    returns: annualised Sortino (higher is better)
    """
    func = gp.compile(individual, pset)
    
    total_sortino = 0.0
    n_folds = len(X_train)
    
    for fold in range(n_folds):
        # Training – not used for fitness directly, but we need to ensure no lookahead
        # The individual is applied to test data
        X_fold = X_test[fold]      # (n_test, n_features)
        y_fold = y_test[fold]      # (n_test,)
        rf_fold = risk_free_rates[fold]  # (n_test,)
        
        # Generate signals (clipped to [0,1])
        signals = np.array([clip_signal(func(*row)) for row in X_fold])
        # Daily returns pct = signal * next_day_return (already aligned)
        # y_fold is next day's return (log return, convert to simple for P&L?)
        # Keep as log return for compounding, but transaction costs are additive in log space?
        # We'll compute simple returns: exp(log_return) - 1
        simple_returns = np.exp(y_fold) - 1.0
        # P&L from holding
        daily_pnl = signals * simple_returns
        # Transaction cost when signal changes
        signal_changes = np.abs(np.diff(np.concatenate(([0], signals))))
        transaction_costs = signal_changes * trans_cost
        # Net log return (approx simple return minus cost)
        net_pnl = daily_pnl - transaction_costs
        # Convert to excess return over risk‑free
        excess = net_pnl - rf_fold   # rf_fold is daily risk‑free simple return
        # Sortino: mean excess / downside std (only negative returns)
        downside = excess[excess < 0].std()
        if downside == 0:
            sortino = 0.0
        else:
            sortino = (excess.mean() / downside) * np.sqrt(252)  # annualised
        total_sortino += sortino
    
    avg_sortino = total_sortino / n_folds
    # Parsimony penalty
    penalty = config.PARSIMONY_COEFF * (individual.height)
    return avg_sortino - penalty,

def run_gp_for_window(ticker, returns_df, macro_df, train_test_pairs):
    """
    Evolve a formula for a single ticker on a fixed historical window.
    train_test_pairs: list of (X_train, y_train, X_test, y_test, rf_train, rf_test)
    """
    # Build feature list: lagged returns + macro
    feature_names = [f"ret_lag_{i}" for i in range(1, config.LOOKBACK_DAYS+1)] + config.MACRO_COLS
    pset = build_primitive_set(feature_names)
    
    # Setup DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=config.INIT_DEPTH_MIN, max_=config.INIT_DEPTH_MAX)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    
    # Evaluation function (uses closure over train_test_pairs, pset)
    def eval_func(individual):
        return evaluate_individual(individual, pset, 
                                   X_train=[p[0] for p in train_test_pairs],
                                   y_train=[p[1] for p in train_test_pairs],
                                   X_test=[p[2] for p in train_test_pairs],
                                   y_test=[p[3] for p in train_test_pairs],
                                   risk_free_rates=[p[4] for p in train_test_pairs],
                                   trans_cost=config.TRANSACTION_COST)
    toolbox.register("evaluate", eval_func)
    toolbox.register("select", tools.selTournament, tournsize=config.TOURNAMENT_SIZE)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, pset=pset, min_=0, max_=config.MAX_DEPTH)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    # Bloat control
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
                        ngen=config.GENERATIONS, stats=stats, halloffame=hof, verbose=True)
    
    best = hof[0]
    best_expr = gp.compile(best, pset)
    # Return the best individual and its string representation
    return best, str(best)
