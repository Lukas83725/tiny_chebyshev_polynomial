"""Chebyshev polynomials of the first and second kind.

Both families are evaluated with the three-term recurrence, which is the
standard way to compute a value without building the polynomial coefficients.
The recurrence is numerically stable for |x| <= 1 and is the same one used by
``numpy.polynomial.chebyshev`` and by the classic Numerical Recipes routines.

We deliberately avoid a closed form (cos(n*acos(x))) because it is inaccurate
near the endpoints and fails for |x| > 1, where the recurrence still works.
"""

from __future__ import annotations

from typing import List

__all__ = [
    "chebyshev_t",
    "chebyshev_u",
    "chebyshev_t_array",
    "chebyshev_u_array",
]


def _validate_n(n: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must be non-negative")


def chebyshev_t(n: int, x: float) -> float:
    """Evaluate the Chebyshev polynomial of the first kind T_n(x).

    Uses the recurrence T_0 = 1, T_1 = x, T_{k+1} = 2x T_k - T_{k-1}.

    Parameters
    ----------
    n : int
        Degree, must be >= 0.
    x : float
        Evaluation point.

    Returns
    -------
    float
        T_n(x).

    Raises
    ------
    TypeError
        If ``n`` is not an integer (booleans are rejected).
    ValueError
        If ``n`` is negative.
    """
    _validate_n(n)
    if n == 0:
        return 1.0
    if n == 1:
        return float(x)

    t_prev = 1.0
    t_curr = float(x)
    for _ in range(2, n + 1):
        t_next = 2.0 * x * t_curr - t_prev
        t_prev, t_curr = t_curr, t_next
    return t_curr


def chebyshev_u(n: int, x: float) -> float:
    """Evaluate the Chebyshev polynomial of the second kind U_n(x).

    Uses the recurrence U_0 = 1, U_1 = 2x, U_{k+1} = 2x U_k - U_{k-1}.

    Parameters
    ----------
    n : int
        Degree, must be >= 0.
    x : float
        Evaluation point.

    Returns
    -------
    float
        U_n(x).

    Raises
    ------
    TypeError
        If ``n`` is not an integer (booleans are rejected).
    ValueError
        If ``n`` is negative.
    """
    _validate_n(n)
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x

    u_prev = 1.0
    u_curr = 2.0 * x
    for _ in range(2, n + 1):
        u_next = 2.0 * x * u_curr - u_prev
        u_prev, u_curr = u_curr, u_next
    return u_curr


def chebyshev_t_array(n: int, xs: List[float]) -> List[float]:
    """Evaluate T_n at every point in ``xs``.

    This is a convenience wrapper. It does not vectorise; it simply calls
    :func:`chebyshev_t` for each element. Kept here so callers do not have to
    write the loop themselves and so the error semantics are identical.

    Parameters
    ----------
    n : int
        Degree, must be >= 0.
    xs : list of float
        Evaluation points.

    Returns
    -------
    list of float
        ``[T_n(x) for x in xs]``.
    """
    _validate_n(n)
    return [chebyshev_t(n, x) for x in xs]


def chebyshev_u_array(n: int, xs: List[float]) -> List[float]:
    """Evaluate U_n at every point in ``xs``.

    See :func:`chebyshev_t_array` for the rationale; the same loop applies.

    Parameters
    ----------
    n : int
        Degree, must be >= 0.
    xs : list of float
        Evaluation points.

    Returns
    -------
    list of float
        ``[U_n(x) for x in xs]``.
    """
    _validate_n(n)
    return [chebyshev_u(n, x) for x in xs]
