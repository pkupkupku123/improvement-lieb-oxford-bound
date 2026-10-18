#!/usr/bin/env python3
"""Exact check of signed two-measure LO certificates; standard library only.

No floating-point calculation participates in a pass/fail decision.
The continuum extension uses linear interpolation in x=1/u and a
piecewise C1 shell kernel with explicitly bounded second derivatives.
"""
import argparse
from fractions import Fraction as Q
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time


def ceildiv(n, d):
    assert d > 0
    return -((-n)//d)


def shell_fraction(a, b, r, s):
    """Exact shell k(a/r,b/s), coordinates have a common scale."""
    rb, sa, ab = r*b, s*a, a*b
    if rb+sa <= ab:
        return 0, 1
    if rb-sa >= ab:
        return r-a, r
    if sa-rb >= ab:
        return s-b, s
    return (rb+sa-ab)**2, 4*r*s*a*b


def curvature_fraction(a, b, c, d, r, s, positive):
    """-signed(k)_xx * (1/a-1/b)^2/8, without the weight."""
    if positive:
        if r*d-s*a < a*d:
            return 0, 1
        return b*(b-a)**2, 4*r*a*a
    if r*c+s*a < a*c or r*c-s*b > b*c or s*a-r*d > a*d:
        return 0, 1
    tn, td = (s-d)**2, d*d
    if (s-c)**2*td > tn*c*c:
        tn, td = (s-c)**2, c*c
    if r*r*td < tn*a*a:
        tn, td = r*r, a*a
    return b*d*(b-a)**2*tn, 16*r*s*a*a*td


def verify(path):
    if not __debug__:
        raise RuntimeError("Do not run with -O")
    start = time.monotonic()
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    assert data["format"] in ("signed-lo-linear-reciprocal-v1",
                              "covariance-lo-linear-reciprocal-v1")
    cs, ms, gs = (data[k] for k in
                  ("coordinate_scale", "mass_scale", "allocation_scale"))
    r, p, q = (data[k] for k in ("radii", "mu", "nu"))
    edges, G = data["edges"], data["allocation"]
    assert all(type(z) is int and z > 0 for z in (cs, ms, gs))
    assert len(r) == len(p) == len(q) > 0
    assert all(type(z) is int for z in r+p+q+edges+G)
    assert all(z > 0 for z in r+edges+G)
    assert all(z >= 0 for z in p)
    assert sum(p) == sum(q) == ms
    assert len(edges) == len(G) >= 2
    assert all(a < b for a, b in zip(edges, edges[1:]))
    assert edges[-1] >= 3*max(r)
    channels = data.get("covariance_channels", [])
    if data["format"] == "signed-lo-linear-reciprocal-v1":
        assert not channels
    for z in channels:
        assert len(z) == len(r) and all(type(t) is int for t in z)
        assert sum(z) == 0
    terms = [(rr, ss, p[i]*q[j]+q[i]*p[j]-q[i]*q[j]
              -sum(z[i]*z[j] for z in channels))
             for i, rr in enumerate(r) for j, ss in enumerate(r)]
    terms = [x for x in terms if x[2]]
    positive_mass = sum(max(w, 0) for _, _, w in terms)
    assert sum(w for _, _, w in terms) == ms*ms
    assert min(G)*ms**2*cs**3 >= 2*positive_mass*edges[0]**3*gs
    ks = 10**18

    @lru_cache(maxsize=100000)
    def upper_kernel(a, b, c, d):
        total = 0
        for rr, ss, w in terms:
            num, den = shell_fraction(a if w > 0 else b,
                                      c if w > 0 else d, rr, ss)
            total += ceildiv(w*num*ks, den*ms**2)
        return total

    @lru_cache(maxsize=None)
    def minimum(l, h):
        return min(G[l:h+1])

    def lhs_sufficient(gi, gj, u, v, numerator, denominator):
        return (cs**3*(gi*u**3+gj*v**3)*denominator
                >= numerator*gs*u**3*v**3)

    def leaf(i, j):
        a, b, c, d = edges[i], edges[i+1], edges[j], edges[j+1]
        error = 0
        for rr, ss, w in terms:
            for aa, bb, cc, dd, ri, sj in (
                    (a, b, c, d, rr, ss), (c, d, a, b, ss, rr)):
                num, den = curvature_fraction(aa, bb, cc, dd, ri, sj, w > 0)
                error += ceildiv(abs(w)*num*ks, den*ms**2)
        eps = max(Q((b-a)**2*(a+2*b), 4*a**3),
                  Q((d-c)**2*(c+2*d), 4*c**3))
        fac = 1+eps
        for ii in (i, i+1):
            for jj in (j, j+1):
                u, v = edges[ii], edges[jj]
                ku = upper_kernel(u, u, v, v)+error
                if not lhs_sufficient(G[ii], G[jj], u, v,
                                      2*fac.numerator*ku, fac.denominator*ks):
                    raise AssertionError(f"Failed curvature cell {(i,j)}, "
                                         f"corner {(ii,jj)}")

    n = len(edges)-1
    pending = [(0, n, 0, n)]
    checked = accepted = second_order = 0
    last_report = start
    while pending:
        l, h, s, t = pending.pop()
        if l >= t:
            continue
        a, b, c, d = edges[l], edges[h], edges[s], edges[t]
        ku = upper_kernel(a, b, c, d)
        checked += 1
        if lhs_sufficient(minimum(l, h), minimum(s, t), b, d, 2*ku, ks):
            accepted += 1
        elif h-l == 1 and t-s == 1:
            leaf(l, s)
            second_order += 1
        elif h-l >= t-s:
            mid = (l+h)//2
            pending.extend([(l, mid, s, t), (mid, h, s, t)])
        else:
            mid = (s+t)//2
            pending.extend([(l, h, s, mid), (l, h, mid, t)])
        if time.monotonic()-last_report > 20:
            print(f"rectangles {checked}, curvature cells {second_order}, "
                  f"pending {len(pending)}", flush=True)
            last_report = time.monotonic()

    costscale = 10**18
    middle = sum(ceildiv((gi+gj)*cs*(b-a)*costscale, 2*gs*a*b)
                 for a, b, gi, gj in zip(edges, edges[1:], G, G[1:]))
    origin = Q(positive_mass*edges[0]**2, 2*ms**2*cs**2)
    tail = Q(27*sum(max(w, 0)*ss**3 for _, ss, w in terms),
             128*ms**2*cs**2*edges[-1])
    integral = origin+Q(middle, costscale)+tail
    energy = sum((Q(pp*ps*cs, 2*ms**2*max(rr, ss))
                  for rr, pp in zip(r, p) for ss, ps in zip(r, p)), Q(0))
    atan5_upper = sum((Q((-1)**j, (2*j+1)*5**(2*j+1))
                       for j in range(5)), Q(0))
    atan239_lower = Q(1, 239)-Q(1, 3*239**3)
    assert 16*atan5_upper-4*atan239_lower < Q(355, 113)
    cube = Q(27, 4)*energy**2*2*Q(355, 113)*integral
    claim = Q(data["claimed_constant"])
    assert 0 < claim and cube <= claim**3, "Claimed constant too small"
    result = dict(status="PASS", certificate_sha256=hashlib.sha256(raw).hexdigest(),
                  claimed_constant_exact=str(claim), shells=len(r), intervals=n,
                  rectangles_checked=checked, monotone_blocks=accepted,
                  curvature_cells=second_order, self_energy_approx=float(energy),
                  integral_upper_approx=float(integral),
                  origin_integral_approx=float(origin), tail_integral_approx=float(tail),
                  constant_upper_approx=float(cube)**(1/3),
                  nu_total_variation_approx=sum(abs(x) for x in q)/ms,
                  additional_covariance_channels=len(channels),
                  seconds=time.monotonic()-start)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", nargs="?", default=str(
        Path(__file__).with_name("lo_signed_certificate.json")))
    parser.add_argument("--result")
    args = parser.parse_args()
    result = verify(args.certificate)
    if args.result:
        Path(args.result).write_text(json.dumps(result, indent=2))
