# szleb_mpcrl/solvers/qdldl_ctypes.py
from __future__ import annotations

import ctypes as ct
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

DEFAULT_TYPES_HEADER = "/usr/local/include/qdldl/qdldl_types.h"


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _find_types_header() -> str:
    env = os.environ.get("QDLDL_TYPES_HEADER", "").strip()
    if env and os.path.isfile(env):
        return env
    if os.path.isfile(DEFAULT_TYPES_HEADER):
        return DEFAULT_TYPES_HEADER
    raise FileNotFoundError(
        "Cannot find qdldl_types.h. Expected /usr/local/include/qdldl/qdldl_types.h "
        "or set QDLDL_TYPES_HEADER=/path/to/qdldl_types.h"
    )


def _parse_typedefs(types_h: str) -> Tuple[str, str]:
    txt = _read_file(types_h)
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
    txt = re.sub(r"//.*", "", txt)

    int_m = re.search(r"typedef\s+([A-Za-z0-9_\s]+?)\s+QDLDL_int\s*;", txt)
    flt_m = re.search(r"typedef\s+([A-Za-z0-9_\s]+?)\s+QDLDL_float\s*;", txt)
    if not int_m or not flt_m:
        raise RuntimeError(f"Could not parse QDLDL_int/QDLDL_float typedefs from {types_h}")

    c_int = " ".join(int_m.group(1).split())
    c_flt = " ".join(flt_m.group(1).split())
    return c_int, c_flt


def _ctype_from_c_name(c_name: str):
    c_name = c_name.strip()

    # Integer types
    if c_name in ("long long", "signed long long"):
        return ct.c_longlong, np.dtype(np.int64)
    if c_name in ("int", "signed int"):
        return ct.c_int, np.dtype(np.int32)
    if c_name in ("long", "signed long"):
        return ct.c_long, np.dtype(np.int64 if ct.sizeof(ct.c_long) == 8 else np.int32)
    if c_name == "int64_t":
        return ct.c_int64, np.dtype(np.int64)
    if c_name == "int32_t":
        return ct.c_int32, np.dtype(np.int32)

    # Float types
    if c_name == "double":
        return ct.c_double, np.dtype(np.float64)
    if c_name == "float":
        return ct.c_float, np.dtype(np.float32)

    raise RuntimeError(f"Unsupported typedef base type '{c_name}'")


def _load_qdldl() -> ct.CDLL:
    env_path = os.environ.get("QDLDL_LIB_PATH", "").strip()
    if env_path:
        return ct.CDLL(env_path)
    return ct.CDLL("libqdldl.so")


@dataclass
class QDLDLTypes:
    IntC: type
    FloatC: type
    BoolC: type
    int_dtype: np.dtype
    float_dtype: np.dtype
    c_int_name: str
    c_float_name: str
    types_header: str


@dataclass
class QDLDLFactor:
    n: int
    Lp: np.ndarray
    Li: np.ndarray
    Lx: np.ndarray
    Dinv: np.ndarray
    types: QDLDLTypes


_LIB: Optional[ct.CDLL] = None
_TYPES: Optional[QDLDLTypes] = None

# exported for the rest of the solver stack
_INT_DTYPE: np.dtype
_FLOAT_DTYPE: np.dtype


def _bind_signatures(lib: ct.CDLL, T: QDLDLTypes) -> None:
    lib.QDLDL_etree.argtypes = [
        T.IntC,
        ct.POINTER(T.IntC),
        ct.POINTER(T.IntC),
        ct.POINTER(T.IntC),
        ct.POINTER(T.IntC),
        ct.POINTER(T.IntC),
    ]
    lib.QDLDL_etree.restype = T.IntC

    lib.QDLDL_factor.argtypes = [
        T.IntC,
        ct.POINTER(T.IntC),
        ct.POINTER(T.IntC),
        ct.POINTER(T.FloatC),
        ct.POINTER(T.IntC),
        ct.POINTER(T.IntC),
        ct.POINTER(T.FloatC),
        ct.POINTER(T.FloatC),
        ct.POINTER(T.FloatC),
        ct.POINTER(T.IntC),
        ct.POINTER(T.IntC),
        ct.POINTER(T.BoolC),
        ct.POINTER(T.IntC),
        ct.POINTER(T.FloatC),
    ]
    lib.QDLDL_factor.restype = T.IntC

    lib.QDLDL_solve.argtypes = [
        T.IntC,
        ct.POINTER(T.IntC),
        ct.POINTER(T.IntC),
        ct.POINTER(T.FloatC),
        ct.POINTER(T.FloatC),
        ct.POINTER(T.FloatC),
    ]
    lib.QDLDL_solve.restype = None


def _resolve_types() -> QDLDLTypes:
    global _LIB, _TYPES, _INT_DTYPE, _FLOAT_DTYPE
    if _TYPES is not None and _LIB is not None:
        return _TYPES

    types_h = _find_types_header()
    c_int_name, c_float_name = _parse_typedefs(types_h)
    IntC, int_dt = _ctype_from_c_name(c_int_name)
    FloatC, flt_dt = _ctype_from_c_name(c_float_name)

    _LIB = _load_qdldl()
    _TYPES = QDLDLTypes(
        IntC=IntC,
        FloatC=FloatC,
        BoolC=ct.c_ubyte,
        int_dtype=int_dt,
        float_dtype=flt_dt,
        c_int_name=c_int_name,
        c_float_name=c_float_name,
        types_header=types_h,
    )
    _bind_signatures(_LIB, _TYPES)

    _INT_DTYPE = int_dt
    _FLOAT_DTYPE = flt_dt
    return _TYPES


def get_qdldl_types() -> QDLDLTypes:
    return _resolve_types()


def _as_contig(a: np.ndarray, dtype: np.dtype) -> np.ndarray:
    a = np.asarray(a, dtype=dtype)
    if not a.flags["C_CONTIGUOUS"]:
        a = np.ascontiguousarray(a)
    return a


def factor_quasidefinite_upper_csc(n: int, Ap: np.ndarray, Ai: np.ndarray, Ax: np.ndarray) -> QDLDLFactor:
    T = _resolve_types()
    lib = _LIB  # type: ignore

    Ap = _as_contig(Ap, T.int_dtype)
    Ai = _as_contig(Ai, T.int_dtype)
    Ax = _as_contig(Ax, T.float_dtype)

    etree = np.empty(n, dtype=T.int_dtype)
    Lnz = np.empty(n, dtype=T.int_dtype)

    iwork = np.empty(3 * n, dtype=T.int_dtype)
    work_et = iwork[:n]
    bwork = np.empty(n, dtype=np.uint8)
    fwork = np.empty(n, dtype=T.float_dtype)

    sumLnz = int(
        lib.QDLDL_etree(
            T.IntC(n),
            Ap.ctypes.data_as(ct.POINTER(T.IntC)),
            Ai.ctypes.data_as(ct.POINTER(T.IntC)),
            work_et.ctypes.data_as(ct.POINTER(T.IntC)),
            Lnz.ctypes.data_as(ct.POINTER(T.IntC)),
            etree.ctypes.data_as(ct.POINTER(T.IntC)),
        )
    )
    if sumLnz < 0:
        raise RuntimeError("QDLDL_etree failed: need upper-tri CSC and >=1 entry per column.")

    Lp = np.empty(n + 1, dtype=T.int_dtype)
    Li = np.empty(sumLnz, dtype=T.int_dtype)
    Lx = np.empty(sumLnz, dtype=T.float_dtype)
    D = np.empty(n, dtype=T.float_dtype)
    Dinv = np.empty(n, dtype=T.float_dtype)

    status = int(
        lib.QDLDL_factor(
            T.IntC(n),
            Ap.ctypes.data_as(ct.POINTER(T.IntC)),
            Ai.ctypes.data_as(ct.POINTER(T.IntC)),
            Ax.ctypes.data_as(ct.POINTER(T.FloatC)),
            Lp.ctypes.data_as(ct.POINTER(T.IntC)),
            Li.ctypes.data_as(ct.POINTER(T.IntC)),
            Lx.ctypes.data_as(ct.POINTER(T.FloatC)),
            D.ctypes.data_as(ct.POINTER(T.FloatC)),
            Dinv.ctypes.data_as(ct.POINTER(T.FloatC)),
            Lnz.ctypes.data_as(ct.POINTER(T.IntC)),
            etree.ctypes.data_as(ct.POINTER(T.IntC)),
            bwork.ctypes.data_as(ct.POINTER(T.BoolC)),
            iwork.ctypes.data_as(ct.POINTER(T.IntC)),
            fwork.ctypes.data_as(ct.POINTER(T.FloatC)),
        )
    )
    if status < 0:
        raise RuntimeError("QDLDL_factor failed (matrix singular / not quasi-definite).")

    return QDLDLFactor(n=n, Lp=Lp, Li=Li, Lx=Lx, Dinv=Dinv, types=T)


def solve_inplace(fact: QDLDLFactor, x: np.ndarray) -> None:
    T = fact.types
    lib = _LIB  # type: ignore

    x = _as_contig(x, T.float_dtype)
    if x.ndim != 1 or x.size != fact.n:
        raise ValueError("x must be shape (n,)")

    lib.QDLDL_solve(
        T.IntC(fact.n),
        fact.Lp.ctypes.data_as(ct.POINTER(T.IntC)),
        fact.Li.ctypes.data_as(ct.POINTER(T.IntC)),
        fact.Lx.ctypes.data_as(ct.POINTER(T.FloatC)),
        fact.Dinv.ctypes.data_as(ct.POINTER(T.FloatC)),
        x.ctypes.data_as(ct.POINTER(T.FloatC)),
    )


def _debug_types_str() -> str:
    T = get_qdldl_types()
    return f"QDLDL typedefs: QDLDL_int={T.c_int_name}, QDLDL_float={T.c_float_name}, header={T.types_header}"


# Resolve typedefs + bind signatures at import time (safe)
_resolve_types()