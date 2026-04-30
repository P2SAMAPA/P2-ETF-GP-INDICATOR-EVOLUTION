import random
import numpy as np
from deap import base, creator, gp, tools, algorithms
import operator
import functools
import config

def protected_div(x, y):
    return x / y if abs(y) > 1e-6 else 1.0

def build_pset(feature_names):
    pset = gp.PrimitiveSet("main", len(feature_names))
    pset.renameArguments(**{f"ARG{i}": name for i, name in enumerate(feature_names)})
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addEphemeralConstant("rand", functools.partial(random.uniform, -1, 1))
    pset.addTerminal(0.0)
    pset.addTerminal(0.5)
    pset.addTerminal(1.0)
    return pset

def evaluate(individual, pset, X_train, y_train, X_test, y_test, rf_test, trans_cost):
    func = gp.compile(individual, pset)
    signals = []
    for row in X_test:
        try:
            v = func(*row)
            signals.append(max(0.0, min(1.0, v)))
        except:
            signals.append(0.5)
    signals = np.array(signals)
    returns = np.exp(y_test) - 1.0
    pnl = signals * returns
    # transaction cost
    changes = np.abs(np.diff(np.concatenate(([0], signals))))
    pnl -= changes * trans_cost
    excess = pnl - rf_test
    if excess.std() == 0:
        sharpe = 0.0
    else:
        sharpe = (excess.mean() / excess.std()) * np.sqrt(252)
    return sharpe,

def run_gp(ticker, X_train, y_train, X_test, y_test, rf_test, feature_names):
    pset = build_pset(feature_names)
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
    toolbox.register("evaluate", lambda ind: evaluate(ind, pset, X_train, None, X_test, y_test, rf_test, config.TRANSACTION_COST))
    toolbox.register("select", tools.selTournament, tournsize=config.TOURNAMENT_SIZE)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, pset=pset, min_=0, max_=config.MAX_DEPTH)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.MAX_DEPTH))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config.MAX_DEPTH))
    pop = toolbox.population(n=config.POPULATION_SIZE)
    hof = tools.HallOfFame(config.HALL_OF_FAME_SIZE)
    algorithms.eaSimple(pop, toolbox, cxpb=config.CROSSOVER_PROB, mutpb=config.MUTATION_PROB,
                        ngen=config.GENERATIONS, halloffame=hof, verbose=False)
    return hof[0], str(hof[0])
