"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable


def mul(x: float, y: float) -> float:
    """f(x, y) = x * y"""
    return x * y


def id(x: float) -> float:
    """f(x) = x"""
    return x


def add(x: float, y: float) -> float:
    """f(x, y) = x + y"""
    return x + y


def neg(x: float) -> float:
    """f(x) = -x"""
    return -x


def lt(x: float, y: float) -> float:
    """Check if x is less than y."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """f(x) = 1.0 if x == y else 0.0"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """f(x) = x if x > y else y"""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """f(x) = |x - y| < 1e-2"""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Compute the sigmoid function.

    Calculate as

    $f(x) =  1.0/(1.0 + e^{-x})$ if x >=0 else $e^x/(1.0 + e^{x})$

    for stability.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Compute the ReLU function.

    f(x) = x if x > 0 else 0
    """
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """f(x) = log(x)"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """f(x) = e^x"""
    return math.exp(x)


def inv(x: float) -> float:
    """f(x) = 1/x"""
    return 1.0 / x


def log_back(x: float, d: float) -> float:
    """Compute the derivative of log(x) over x."""
    return d / (x + EPS)


def inv_back(x: float, d: float) -> float:
    """Compute the derivative of d/x over x."""
    return -(1.0 / x**2) * d


def relu_back(x: float, d: float) -> float:
    """Compute the derivative of the d*relu(x) over x."""
    return d if x > 0 else 0.0


def map(f: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order function that applies a given function to each element of an iterable and returns a list of results."""

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(f(x))
        return ret

    return _map


def negList(x: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map"""
    return map(neg)(x)


def zipWith(
    f: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines elements from two iterables using a given function. It only performs on the minimum length of the two iterables."""

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(f(x, y))
        return ret

    return _zipWith


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith"""
    return zipWith(add)(ls1, ls2)


def reduce(
    f: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    """Higher-order function that reduces an iterable to a single value using a given function."""

    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for i in ls:
            val = f(val, i)
        return val

    return _reduce


def sum(x: Iterable[float]) -> float:
    """Sum all elements in a list using reduce"""
    return reduce(add, 0.0)(x)


def prod(x: Iterable[float]) -> float:
    """Calculate the product of all elements in a list using reduce"""
    return reduce(mul, 1.0)(x)
