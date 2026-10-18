"""Fast exact arithmetic and historical-receipt audit; NOT a new domain proof.

Only the Python standard library is needed. Integer scales in the certificates
are authoritative. Floating point is used only for human-readable diagnostics.
"""
import csv
import gzip
import hashlib
import json
from decimal import Decimal, ROUND_CEILING
from fractions import Fraction as Q
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    """Use explicit failures so quick checks also work with Python -O."""
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def ceil_fraction(value, scale=10**18):
    # Same outward rounding as the exact verifier; never round downward.
    return Q(-((-value.numerator * scale) // value.denominator), scale)


def upward5(value):
    return str(Decimal(str(value)).quantize(Decimal('0.00001'), rounding=ROUND_CEILING))


def check_channel(m, provenance):
    directory = ROOT / 'data' / f'm{m}'
    meta = provenance[str(m)]
    objects = {}
    for name, expected in meta['original_files_sha256'].items():
        path = directory / name
        raw = gzip.decompress(path.with_suffix('.json.gz').read_bytes()) if name in (
            'verified.json', 'audit.json') else path.read_bytes()
        require(sha(raw) == expected, f'm{m}: checksum mismatch: {name}')
        objects[name] = json.loads(raw)
    d, receipt, audit = (objects[k] for k in ('certificate.json', 'verified.json', 'audit.json'))
    cs, ms, gs = (d[k] for k in ('coordinate_scale', 'mass_scale', 'allocation_scale'))
    r, p, zs, edges, h = (d[k] for k in ('radii', 'mu', 'covariance_channels', 'edges', 'allocation'))
    require(d['format'] == 'covariance-lo-linear-reciprocal-v1', 'Wrong certificate format')
    require(all(type(x) is int and x > 0 for x in (cs, ms, gs)), 'Invalid scales')
    require(len(zs) == m and d['nu'] == p, 'Wrong channel representation')
    require(len(r) == len(p) and all(len(z) == len(r) for z in zs), 'Invalid dimensions')
    require(all(type(x) is int for row in [r, p, edges, h] + zs for x in row), 'Noninteger data')
    require(min(r) > 0 and min(p) >= 0 and sum(p) == ms, 'Invalid mu')
    require(all(sum(z) == 0 for z in zs), 'Channel mass is not zero')
    require(len(edges) == len(h) == 6401 and min(h) > 0 and edges[0] > 0, 'Invalid nodes')
    require(all(a < b for a, b in zip(edges, edges[1:])), 'Unsorted nodes')
    require(all(a < b for a, b in zip(r, r[1:])), 'Unsorted radii')
    measures = objects['measures.json']
    index = {v: i for i, v in enumerate(r)}
    require(len(measures['radii']) == len(measures['weights']) == m+1, 'Invalid measures')
    for rr, ww, target in zip(measures['radii'], measures['weights'], [p] + zs):
        require(len(rr) == len(ww) == 100, 'Expected 100 parameters per measure')
        mapped = [0] * len(r)
        for radius, weight in zip(rr, ww):
            mapped[index[radius]] += weight
        require(mapped == target, 'Independent-support mapping mismatch')

    # Compute W in integer units ms^2, avoiding a dense matrix of Fractions.
    positive = moment = total = 0
    for i, pi in enumerate(p):
        row_sum = 0
        for j, pj in enumerate(p):
            w = pi*pj - sum(z[i]*z[j] for z in zs)
            row_sum += w
            if w > 0:
                positive += w
                moment += w*r[j]**3
        require(row_sum == pi*ms, 'W row-sum identity failed')
        total += row_sum
    require(total == ms**2, 'W total-mass identity failed')
    require(min(h)*ms**2*cs**3 >= 2*positive*edges[0]**3*gs, 'Origin floor failed')
    require(edges[-1] >= 3*max(r), 'Tail cutoff failed')
    origin = Q(positive*edges[0]**2, 2*ms**2*cs**2)
    tail = Q(27*moment, 128*ms**2*cs**2*edges[-1])
    # Outward-round each exact trapezoid to a common 10^-18 denominator.
    # This is an upper bound on the exact integral, not numerical quadrature.
    middle = sum((ceil_fraction(Q((u+v)*cs*(b-a), 2*gs*a*b))
                  for a, b, u, v in zip(edges, edges[1:], h, h[1:])), Q(0))
    integral = origin + middle + tail
    active = [(ri, pi) for ri, pi in zip(r, p) if pi]
    energy = sum((Q(pi*pj*cs, 2*ms**2*max(ri, rj))
                  for ri, pi in active for rj, pj in active), Q(0))
    # Machin's formula and alternating-series bounds justify pi < 355/113.
    atan5 = sum((Q((-1)**j, (2*j+1)*5**(2*j+1)) for j in range(5)), Q(0))
    atan239 = Q(1, 239) - Q(1, 3*239**3)
    require(16*atan5 - 4*atan239 < Q(355, 113), 'Pi bound failed')
    cube = Q(27, 2)*Q(355, 113)*energy**2*integral
    claim = Q(d['claimed_constant'])
    require(0 < cube <= claim**3, 'Final rational cube inequality failed')
    for key, value in [('energy_exact', energy), ('integral_upper_exact', integral), ('cube_upper_exact', cube)]:
        require(Q(audit[key]) == value, f'Historical audit mismatch: {key}')

    verifier_sha = sha((ROOT / 'code/verification/verify_signed_lo_certificate.py').read_bytes())
    dispatcher_sha = sha((ROOT / 'code/verification/verify_parallel.py').read_bytes())
    require(receipt['original_verifier_sha256'] == verifier_sha, 'Verifier version mismatch')
    require(receipt['parallel_verifier_sha256'] == dispatcher_sha, 'Dispatcher version mismatch')
    require(receipt['status'] == audit['status'] == 'PASS', 'Historical verification did not pass')
    require(receipt['certificate_sha256'] == audit['certificate_sha256'] == meta['sha256'], 'Receipt mismatch')
    require(audit['full_domain_coverage'] and audit['algebraic_channel_rank'] == m, 'Invalid historical audit')
    require(receipt['claimed_constant_exact'] == str(claim), 'Claim mismatch')
    require(receipt['self_energy_approx'] == float(energy) and receipt['integral_upper_approx'] == float(integral), 'Totals mismatch')
    cursor = 0
    for part in receipt['shard_results']:
        lo, hi = part['first_coordinate_band']
        require(lo == cursor and lo < hi <= 6400, 'Gap/overlap in historical coverage')
        cursor = hi
        require(part['status'] == 'PASS' and part['certificate_sha256'] == meta['sha256'], 'Invalid shard')
        require(part['additional_covariance_channels'] == m and part['intervals'] == 6400, 'Shard dimensions')
        for field in ('claimed_constant_exact', 'self_energy_approx', 'integral_upper_approx',
                      'origin_integral_approx', 'tail_integral_approx', 'constant_upper_approx',
                      'nu_total_variation_approx', 'shells'):
            require(part[field] == receipt[field], f'Shard mismatch: {field}')
    require(cursor == 6400 and len(receipt['shard_results']) == receipt['bands'], 'Incomplete receipt')
    for field in ('rectangles_checked', 'monotone_blocks', 'curvature_cells'):
        require(receipt[field] == sum(x[field] for x in receipt['shard_results']), 'Count mismatch')
    if m == 1:
        nu = [a+b for a, b in zip(p, zs[0])]
        require(sum(nu) == ms and sum(x != 0 for x in nu) == 180 and sum(x < 0 for x in nu) == 64,
                'Signed nu count/mass mismatch')
        require(all(p[i]*p[j]-zs[0][i]*zs[0][j] == p[i]*nu[j]+nu[i]*p[j]-nu[i]*nu[j]
                    for i in range(len(p)) for j in range(len(p))), 'Signed nu identity failed')
    numerical = objects['input_profile.json']['validation_constant']
    expected = {1: ('1.54457', '1.54471'), 2: ('1.53928', '1.53950'),
                3: ('1.53484', '1.53490'), 5: ('1.53413', '1.53418')}
    require((upward5(numerical), upward5(d['claimed_constant'])) == expected[m], 'Table II mismatch')
    return dict(channels=m, numerical_objective=numerical, numerical_upward_5dp=upward5(numerical),
                verified_upper_bound=d['claimed_constant'], verified_upward_5dp=upward5(d['claimed_constant']),
                energy=float(energy), integral_upper=float(integral), certificate_sha256=meta['sha256'],
                check_scope='exact totals and archived receipts; no fresh full-domain verification')


def main():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        require(sha((ROOT / name).read_bytes()) == expected, f'Release checksum mismatch: {name}')
    provenance = json.loads((ROOT / 'data/provenance.json').read_text(encoding='utf-8'))
    rows = [check_channel(m, provenance) for m in (1, 2, 3, 5)]
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    with (output / 'table_II.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'quick_check.json').write_text(json.dumps(rows, indent=2)+'\n', encoding='utf-8')
    for row in rows:
        print(f"m={row['channels']}: numerical {row['numerical_upward_5dp']}; verified {row['verified_upward_5dp']}; exact claim {row['verified_upper_bound']}")
    print('PASS: release hashes, exact totals, historical coverage, Table II and signed-nu identity.')
    print('This quick audit does not rerun the full-domain inequalities.')


if __name__ == '__main__':
    main()
