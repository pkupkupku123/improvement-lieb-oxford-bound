# Release-preparation validation

Prepared on 19 October 2026 with CPython 3.12.1 on Windows x64, NumPy 1.26.4,
SciPy 1.12.0, Numba 0.61.2, llvmlite 0.44.0 and Matplotlib 3.8.3.

Completed checks:

- All 16 Python sources parse successfully.
- Original frozen scientific files match their original manifest hashes;
  decompressed receipts/audits preserve the original bytes.
- All four published certificates pass the new quick audit: exact mass and
  support identities, endpoint conditions, rational energy/integral/cube totals,
  archived full coverage, manuscript Table II rounding and signed-ν identity.
- Figure 1 was regenerated in PDF/SVG/PNG; the PNG was visually inspected.
- A small synthetic certificate passed both serial and parallel full-domain
  verification; a deliberately invalid constant was rejected.
- Numerical grid-LP evaluation ran successfully for m=1,2,3,5 at 64 intervals.
- A five-channel SLSQP restart ran at 32 intervals, one round, two iterations.
  It reached the intentionally small iteration limit and rejected the proposed
  update on the denser validation grid; retaining the starting profile is the
  expected behavior of this execution test, not an optimization improvement.
- The one-channel builder ran at 32 focused intervals. Its deliberately loose
  candidate (C=118.316841632) passed the original continuous exact verifier,
  exercising 172 rectangles and 11 curvature cells. This validates execution
  of the builder-to-verifier path, not the paper's sharper constant.
- The root quick audit was also run with an unrelated current working directory.

The 3520-interval numerical evaluation was started and stopped before completion
because it is a substantially longer calculation. It is **not counted as a
completed release test**. The four 6400-interval paper certificates were **not
fully reverified** during packaging. Their original complete verification
receipts are included, their hashes and internal consistency were checked, and
the unchanged exact verifier is provided for fresh full verification.

`validation_results.json` contains the completed numerical smoke outputs and
the coarse-certificate verification result. These outputs are clearly separate
from the authoritative published data in `data/`. No Linux/macOS execution was
performed during preparation; paths and commands are portable Python, and no
Windows-specific runtime path is needed.

One integration issue was fixed in the public entry point: the original search
CLI's default minimum log-radius gap was 10^-4, but the final two-, three- and
five-channel profiles used 10^-8. `reproduce.py optimize` now reads the saved
parameter constraint, so restarting those profiles does not reject the initial
point as infeasible. The underlying numerical algorithms and exact-verifier
sources were not changed.
