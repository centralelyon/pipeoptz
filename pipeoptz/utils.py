import numpy as np
import random as rd
from math import comb

def product(*iterables, random=False, max_combinations=0, optimize_memory=False):
    """
    Returns the cartesian product of input iterables, with an option for random sampling.

    Args:
        *iterables: Variable number of iterables to compute the product.
        random (bool): If True, returns a random sample from the product instead of all combinations.
        max_combinations (int): The maximum number of combinations to sample.
        optimize_memory (bool): Have an effect only if random is True and max_combinations > 0. 
            If True, optimizes memory usage by generating a random product
            without storing all combinations in memory. But  there is a risk of generating the same 
            value multiple times. Put to True only if max_combinations << len(all_combinations) or if there is no problem
            if the same value is repeated.

    Yields:
        Tuples representing the cartesian product of the input iterables.
    """
    len_index = [len(iterable) for iterable in iterables]
    max_combinations = max_combinations if max_combinations > 0 else np.prod(len_index)

    if random and optimize_memory:
        for i in range(max_combinations):
            yield tuple(it[rd.randrange(length)] for it, length in zip(iterables, len_index))
        return
    
    from itertools import product as it_product
    if random:
        rd_index = list(it_product(*[range(length) for length in len_index]))
        rd.shuffle(rd_index)
        for i in range(min(max_combinations, len(rd_index))):
            yield tuple(iterables[j][rd_index[i][j]] for j in range(len(iterables)))
        return
    
    prod = it_product(*iterables)
    for i in range(min(max_combinations, np.prod(len_index))):
        yield next(prod)

def ith_subset(n: int, i: int) -> list[int]:
    """
    Returns the i-th subset of A = [0, n-1], ordered first by cardinality,
    then lexicographically within each cardinality class.

    Args:
        n (int): Upper bound of the interval A = [0, n-1].
        i (int): Index (0 <= i < 2^n) in the cardinality-sorted power set.

    Returns:
        list[int]: The i-th subset under the cardinality-lex order.
    """
    total = 2**n
    if i < 0 or i >= total:
        raise ValueError(f"Index i must be in [0, {total - 1}]")

    # Step 1: find the cardinality group (number of elements in subset)
    remaining = i
    for k in range(n + 1):  # cardinalities from 0 to n
        c = comb(n, k)
        if remaining < c:
            cardinality = k
            break
        remaining -= c

    # Step 2: generate the `remaining`-th k-combination in lex order
    subset = []
    x = 0
    for j in range(cardinality):
        while comb(n-1 - x, cardinality - j - 1) <= remaining:
            remaining -= comb(n-1 - x, cardinality - j - 1)
            x += 1
        subset.append(x)
        x += 1

    return subset