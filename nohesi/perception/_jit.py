"""The numba switch for perception's compiled kernels (user-approved 2026-11-19).

`njit` compiles with numba (nopython, cached on disk so that worker processes load the compiled code instead of
compiling it again) when numba is installed and the environment variable NOHESI_JIT is not '0'. Otherwise it returns
the function unchanged, so every kernel also runs as plain Python: the reference the tests compare the compiled code
against, and the fallback without numba. Callers choose the compiled or the numpy path with ENABLED.
"""
import os

try:
    import numba
except ImportError:            # pragma: no cover - numba is an optional dependency (pyproject: perception)
    numba = None

ENABLED = numba is not None and os.environ.get('NOHESI_JIT', '1') != '0'

__all__ = ['ENABLED', 'njit']


def njit(fn=None, **kw):
    """numba.njit(cache=True, **kw) when ENABLED, else the identity. Usable bare (@njit) or with options."""
    def wrap(f):
        if not ENABLED:
            return f
        return numba.njit(cache=True, **kw)(f)
    return wrap if fn is None else wrap(fn)
