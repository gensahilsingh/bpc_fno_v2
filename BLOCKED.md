# BLOCKED: Myokit requires C compiler for ODE simulation

## Issue
Myokit's `Simulation` class compiles the TT2006 ionic model to C code at runtime for performance. This requires Microsoft Visual C++ 14.0+ (Build Tools) on Windows.

## Error
```
myokit._err.CompilationError: Unable to compile.
error: Microsoft Visual C++ 14.0 or greater is required.
```

## What was tried
1. Pipeline rewritten to use real TT2006 + MonodomainSolver (no eikonal approximation)
2. CellML model loading works fine (model validation passes)
3. Conductance parameter name mapping fixed
4. Simulation creation fails at compile step

## Options to resolve

### Option A: Install MSVC Build Tools locally (Windows)
Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
Install "Desktop development with C++" workload.
Then rerun: `python scripts/generate_synthetic.py --n-samples 3`

### Option B: Generate data on cloud Linux instance (recommended)
See CLOUD_DATAGEN.md for full instructions.
Linux has gcc by default, so Myokit compiles without issues.
This is the recommended path since data generation is CPU-intensive
(~minutes per sample × 4000 samples).

### Option C: Use WSL2 on this Windows machine
If WSL2 is available, install Python + dependencies in WSL2 and run
data generation there. gcc is available by default on Ubuntu/WSL2.
