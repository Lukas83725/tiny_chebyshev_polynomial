# tiny_chebyshev_polynomial

Evaluates Chebyshev polynomials of the first and second kind via the
three-term recurrence, for approximation and interpolation work where you
want the values without pulling in a numerical library.

## Usage

```python
from tiny_chebyshev_polynomial import chebyshev_t, chebyshev_u

# T_3(0.5) = 4*(0.5)^3 - 3*(0.5) = -1.0
print(chebyshev_t(3, 0.5))   # -1.0

# U_3(0.5) = 8*(0.5)^3 - 4*(0.5) = -1.0
print(chebyshev_u(3, 0.5))   # -1.0
```

The package exports exactly four names:

- `chebyshev_t(n, x) -> float`
- `chebyshev_u(n, x) -> float`
- `chebyshev_t_array(n, xs) -> list[float]`
- `chebyshev_u_array(n, xs) -> list[float]`

`n` must be a non-negative `int`; anything else raises `TypeError` (booleans
are rejected) or `ValueError`. `x` is any real number, including values
outside `[-1, 1]`.

## Why this exists

The Chebyshev polynomials show up constantly in approximation theory and
spectral methods, but pulling in NumPy just to evaluate `T_n(x)` is heavy.
This library is the smallest thing that does the job correctly: it uses the
three-term recurrence

```
T_0 = 1, T_1 = x, T_{k+1} = 2x T_k - T_{k-1}
U_0 = 1, U_1 = 2x, U_{k+1} = 2x U_k - U_{k-1}
```

which is stable for `|x| <= 1` and still works for `|x| > 1`, unlike the
closed form `cos(n * acos(x))` which loses precision near the endpoints and
returns `nan` outside the unit interval. The trade-off is that the recurrence
is `O(n)` per evaluation and does not cache anything; for repeated evaluation
at many `n` you would want a coefficient table, which this library does not
provide.

## Edge cases

The recurrence is well defined for any real `x`, so `chebyshev_t` and
`chebyshev_u` accept values outside `[-1, 1]` and return the analytic
continuation. The one thing to watch for is that `n` must be a real Python
`int`; passing `1.0` or `True` raises `TypeError` rather than silently
rounding, because the recurrence is defined only for non-negative integer
degrees.

## Tests

```
PYTHONPATH=src python -m unittest discover -s tests
```
