# Data format and mathematical conventions

The authoritative objects are the unchanged `data/m*/certificate.json` files.
Their format is `covariance-lo-linear-reciprocal-v1`. All dimensional values
are represented by integer numerators and positive integer scales:

| JSON field | Interpretation |
|---|---|
| `coordinate_scale` | denominator for all radii and interpolation nodes |
| `mass_scale` | denominator for μ and channel weights |
| `allocation_scale` | denominator for h node values |
| `radii` | sorted union of all measures' radii, in coordinate-scale units |
| `mu` | nonnegative μ weights, in mass-scale units; exact sum = mass_scale |
| `covariance_channels[l]` | σ_(l+1) weights on the common support; exact sum = 0 |
| `nu` | storage placeholder equal to `mu`, not the signed auxiliary ν |
| `edges` | 6401 increasing nodes L_i, in coordinate-scale units |
| `allocation` | h(L_i), in allocation-scale units |
| `claimed_constant` | decimal string interpreted as an exact rational number |

`measures.json` retains 100 radius/weight parameters for μ and each channel
before mapping onto the common support; it uses the certificate's scales.
Some parameters have zero weight. `input_profile.json` instead stores ordinary
floating-point radii/weights and the original optimization diagnostics.

## Appendix C mapping

Write r_i, p_i and z_li for the decoded rational values. The effective kernel
coefficient matrix is W_ij = p_i p_j − Σ_l z_li z_lj. Because each channel has
zero mass, Σ_j W_ij = p_i and Σ_ij W_ij = 1.

The verifier's `shell_fraction(a,b,r,s)` implements the manuscript's shell
kernel with integer coordinates sharing the same scale. It returns an exact
numerator and denominator. `curvature_fraction` bounds the signed second
derivative contribution times the squared reciprocal interval width / 8.
The `upper_kernel`, `leaf` and subdivision loop implement the monotonicity and
four-corner interpolation checks.

The control function is h(a)=2f(a)/a³. Let P=Σ_ij max(W_ij,0) and
H=(27/128)Σ_ij max(W_ij,0)r_j³. On the whole domain:

- h(a)=P a³ for 0<a<L_0;
- h is linear in 1/a between successive nodes;
- h(a)=H for a>L_n; f(0)=0.

The weighted integral A is

```text
A = P L_0²/2 + H/L_n
    + Σ_i (h_i+h_(i+1))/2 × (1/L_i − 1/L_(i+1)).
D = 1/2 Σ_ij p_i p_j / max(r_i,r_j).
J[f] = 2π A.
C³ ≤ (27/2) π D² A ≤ (27/2) (355/113) D² A_upper.
```

The integral formula is exact. For efficient arithmetic, the implementation
rounds each rational middle-interval contribution **up** to denominator 10^18,
then adds the exact origin and tail contributions. Consequently `integral_upper`
is a rigorous upper bound, not the unrounded exact integral. The pi inequality
is checked through rational alternating-series bounds in Machin's formula.
Every final comparison is rational; float diagnostics do not decide acceptance.

## Single-channel equivalence and Figure 1

For m=1, reconstruct q_i=p_i+z_1i with integers before converting to floats.
Then p_i q_j+q_i p_j−q_i q_j = p_i p_j−z_1i z_1j exactly. The signed ν has
180 nonzero weights, including 64 negative weights; its total mass is one.

Figure 1(a,c) plots **discrete shell masses**, not continuous volume densities.
No division by shell area or shell spacing is applied. Figure 1(b) uses the
final certified h nodes and the full-domain extensions, not the earlier
floating-point LP allocation. Plotting conversions to floats do not constitute
new verification.

## Provenance and archived receipts

`data/provenance.json` maps each result to its historical source package and
the SHA-256 of every original scientific file. Source paths are documentary;
they are not runtime dependencies. Gzip decompression returns exactly the
original `verified.json` and `audit.json` bytes. The receipts contain every
contiguous first-coordinate band and its aggregate verifier results. Historical
text logs are omitted to avoid duplicating the structured results; hashes of
those omitted logs remain in the original audit as historical metadata.

`MANIFEST.json` covers `data/`, `code/`, `reproduce.py` and `requirements.txt`.
Documentation/citation metadata may be edited without invalidating the
scientific checks; generated `results/` are excluded. `.gitattributes` disables newline conversion, since byte changes
would invalidate the audited verifier hash and certificate provenance.
