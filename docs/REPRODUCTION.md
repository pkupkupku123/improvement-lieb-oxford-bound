# Reproduction workflow and code guide

Run commands from the package root. `reproduce.py` also works by absolute path
from another working directory. All outputs are under the package's `results/`.

## Computational stages

1. `model.py` evaluates the shell kernel with prefix sums, constructs the LP
   in h node values, evaluates its objective and contracts the LP dual with
   parameter derivatives. Radius parameters are logarithmic; the outer μ
   parameter radius is fixed to one.
2. `active_model.py` solves the same grid LP by constraint generation. Every
   omitted row is scanned before acceptance; numerical difficulty falls back
   to the complete LP. This remains a finite-grid numerical calculation.
3. `search.py` / `refine.py` call SLSQP on independent radii and weights, enforce
   μ positivity and channel zero-mass constraints, and validate candidate steps
   on a grid with twice as many intervals. Results are explicitly numerical.
4. `channel_certificate.py` quantizes parameters, constructs interpolation
   margins, solves another LP for h, and rounds the allocation upward. Shell
   kernel/curvature helpers are in `signed_lo/build_certificate.py`;
   `psd_lo/fast_shell_matrices.py` accelerates matrix construction;
   `psd_lo/build_covariance_certificate.py` supplies exact total evaluation.
   Other imported functions are retained only to close these dependencies.
5. `verification/verify_signed_lo_certificate.py` independently checks all
   continuous cells and endpoint conditions with integer/rational arithmetic.
   The parallel dispatcher changes only the initial first-coordinate band in
   the verifier AST and checks complete contiguous coverage of all bands.

The audited verifier and dispatcher are byte-identical to the original sources.
Numerical sources are the final five-channel campaign versions, which accept
all four published channel counts. The shared model and active-LP source
hashes match the hashes recorded in the final profiles. Older search versions
had different allowed channel counts/CLI settings. `data/source_versions.json`
records the selected source versions; no historic optimizer run is claimed
to have been recreated bit-for-bit.

## Grid choices

The optimization grid has 1760 intervals, the final numerical evaluation grid
has 3520, and the continuous certificate has 6400. Optimization/evaluation
nodes are logarithmic over [10^-4,10^5]. The final certificates use a focused
piecewise logarithmic grid with breakpoints
`[10^-4, 0.015, 0.05, 3, 15, 10^5]` and interval fractions `[2,1,10,1,2]/16`.
The exact integer nodes in the certificate, not a regenerated float grid, are
authoritative for verification.

## Explicit commands for new candidates

Refine a profile (choose the saved minimum log-radius gap for that profile):

```sh
python code/multichannel_lo/refine.py --input data/m5/input_profile.json --output results/new_m5.json --stages 1760 --rounds 6 --iterations 40 --threads 2 --minimum-log-gap 1e-8
```

Build a candidate from the resulting profile:

```sh
python code/multichannel_lo/channel_certificate.py --input results/new_m5.json --output results/new_m5_certificate.json --intervals 6400 --threads 2 --fast-matrices --grid-kind focused
```

Verify that candidate, rather than the frozen paper certificate:

```sh
python code/verification/verify_parallel.py results/new_m5_certificate.json --verifier code/verification/verify_signed_lo_certificate.py --result results/new_m5_verified.json --workers 8 --bands 6400
```

A single-process alternative is:

```sh
python code/verification/verify_signed_lo_certificate.py results/new_m5_certificate.json --result results/new_m5_verified_serial.json
```

Failure must be investigated; it must never be converted into a certified
claim by changing tolerances or ignoring the return code. Changing the
optimization or candidate-builder floating-point settings is allowed, but
acceptance still requires the independent exact verifier.

## Small execution example

This runs the builder and the real continuous verifier at inexpensive grid
size. Its very loose bound is an execution check, not a paper result:

```sh
python reproduce.py build --channels 1 --intervals 32 --threads 2
python code/verification/verify_signed_lo_certificate.py results/proposed_m1.json --result results/coarse_verification_m1.json
```

## Dependencies and resource use

The exact verifier, receipt audit, CLI and synthetic smoke test use only the
standard library. Plotting uses NumPy and Matplotlib. Optimization and candidate
construction additionally use SciPy/HiGHS, Numba and llvmlite. Numba's first run
includes JIT compilation and writes caches; these are ignored by Git.

`--threads` controls Numba numerical work; the LP is configured with one HiGHS
thread. `--workers` controls independent exact-verifier processes. Full grid
arrays scale quadratically in the node count. Avoid running several full-size
numerical calculations concurrently on a memory-limited workstation.

The dispatcher does not implement checkpoint resume: each invocation starts a
new full verification. Historical receipts are included so inspecting results
does not require a long rerun. Checksum validation establishes byte identity
and internal consistency, not a new proof of every continuous inequality.
