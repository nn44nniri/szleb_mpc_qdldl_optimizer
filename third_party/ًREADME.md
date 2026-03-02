# Install QDLDL first (required)

QDLDL is a C library. You must build and install the shared library so Python can load it.

Option A (recommended): build & install system-wide
# 1) get QDLDL source
```bash
git clone https://github.com/osqp/qdldl.git
cd qdldl
```

# 2) build shared library
```bash
mkdir -p build && cd build
cmake .. -DQDLDL_BUILD_SHARED_LIB=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

# 3) install
```bash
sudo cmake --install .
```

# 4) ensure loader sees it
```bash
sudo ldconfig
```

This produces something like libqdldl.so in /usr/local/lib and headers in /usr/local/include.

Why this is the right backend: OSQP’s approach solves a quasi-definite KKT linear system repeatedly and benefits from factorization caching; their direct linear solver is QDLDL 

and they explicitly report using QDLDL in benchmarks 

.

# Option B: local build + environment variable (no sudo)
```bash
git clone https://github.com/osqp/qdldl.git
cd qdldl
mkdir -p build && cd build
cmake .. -DQDLDL_BUILD_SHARED_LIB=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

# export for runtime

```bash
export QDLDL_LIB_PATH="$(pwd)/libqdldl.so"
```
Our Python loader below supports QDLDL_LIB_PATH.




## Test:

```bash
source /home/shabgard/Desktop/Optimiser/mpcrl-greenhouse/code/env/bin/activate
cd /home/shabgard/Desktop/Optimiser/mpcrl-greenhouse/code/szleb_mpc_optimizer

python - <<'PY'
import numpy as np
import scipy.sparse as sp
from szleb_mpcrl.solvers.qdldl_ctypes import (
    factor_quasidefinite_upper_csc, solve_inplace,
    _debug_types_str, _INT_DTYPE, _FLOAT_DTYPE
)
from szleb_mpcrl.solvers.csc_utils import to_upper_csc

print(_debug_types_str())
print("INT dtype:", _INT_DTYPE, "FLOAT dtype:", _FLOAT_DTYPE)

A = sp.csc_matrix(np.array([[2.0, 1.0],[1.0, -1.0]], dtype=np.float64))
U = to_upper_csc(A)

Ap = U.indptr.astype(_INT_DTYPE, copy=False)
Ai = U.indices.astype(_INT_DTYPE, copy=False)
Ax = U.data.astype(_FLOAT_DTYPE, copy=False)

F = factor_quasidefinite_upper_csc(2, Ap, Ai, Ax)
b = np.array([1.0, 0.0], dtype=_FLOAT_DTYPE)
solve_inplace(F, b)
print("solution x =", b)
PY
```

* -----------------------------------------------------