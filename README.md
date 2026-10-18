# An Improvement on the Lieb–Oxford Bound

Data and code for Ye Li's manuscript (19 October 2026, Submitted to Phys. Rev. A), supporting Table II,
Figure 1, and Appendix C. The verified bound is **C_LO ≤ 1.5342**.

## Quick start

Use Python 3.10–3.12. Run from this directory:

```sh
python reproduce.py
```

This checks the data, exact arithmetic, and archived verification records using
only the Python standard library. It does not repeat full-domain verification.

To reproduce Figure 1:

```sh
python -m pip install -r requirements.txt
python reproduce.py figure
```

To rerun full-domain verification (a long CPU calculation):

```sh
python reproduce.py verify --channels 5 --workers 8
```

Supported channel counts are `1`, `2`, `3`, and `5`. Outputs go to `results/`.
Do not run verification with `python -O`.

## Files and documentation

- `data/`: optimized parameters, rational certificates, and verification records.
- `code/`: optimization, certificate construction, verification, and plotting.
- [Data format](docs/DATA_FORMAT.md) · [Reproduction commands](docs/REPRODUCTION.md) · [Validation scope](docs/VALIDATION.md)
- [Citation](CITATION.cff)
