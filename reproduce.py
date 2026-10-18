#!/usr/bin/env python3
"""Portable entry point. Default: fast standard-library audit of the paper data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(*arguments):
    subprocess.run([sys.executable, '-B', *map(str, arguments)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='check', choices=[
        'check', 'figure', 'evaluate', 'optimize', 'build', 'verify', 'smoke'])
    parser.add_argument('--channels', type=int, default=5, choices=[1, 2, 3, 5])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--intervals', type=int, help='Defaults: evaluate 3520; optimize 1760; build 6400')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--bands', type=int, default=6400)
    args = parser.parse_args()
    if min(args.threads, args.workers, args.rounds, args.iterations, args.bands) < 1:
        parser.error('Counts must be positive')
    if args.intervals is not None and args.intervals < 16:
        parser.error('Use at least 16 intervals')
    if args.action in ('optimize', 'build', 'verify', 'smoke') and not __debug__:
        parser.error('Do not use Python -O: scientific code uses assertions')
    profile = ROOT / 'data' / f'm{args.channels}' / 'input_profile.json'
    certificate = profile.with_name('certificate.json')
    out = ROOT / 'results'
    out.mkdir(exist_ok=True)
    if args.action == 'check':
        run(ROOT / 'code/check_data.py')
    elif args.action == 'figure':
        run(ROOT / 'code/plot_profiles.py')
    elif args.action == 'evaluate':
        run(ROOT / 'code/evaluate_profile.py', '--input', profile,
            '--output', out / f'evaluation_m{args.channels}.json',
            '--intervals', args.intervals or 3520, '--threads', args.threads)
    elif args.action == 'optimize':
        # Warm restart from published parameters; this is a local SLSQP search.
        saved = json.loads(profile.read_text())
        # Later searches allowed closer radii than the historical CLI default.
        # Preserve that saved constraint so the initial profile is feasible.
        gap = saved.get('parameter_bounds', {}).get('minimum_log_radius_gap', 1e-8)
        run(ROOT / 'code/multichannel_lo/refine.py', '--input', profile,
            '--output', out / f'optimized_m{args.channels}.json', '--stages', args.intervals or 1760,
            '--rounds', args.rounds, '--iterations', args.iterations, '--threads', args.threads,
            '--minimum-log-gap', gap)
    elif args.action == 'build':
        provenance = json.loads((ROOT / 'data/provenance.json').read_text())
        run(ROOT / 'code/multichannel_lo/channel_certificate.py', '--input', profile,
            '--output', out / f'proposed_m{args.channels}.json', '--intervals', args.intervals or 6400,
            '--threads', args.threads, '--fast-matrices',
            '--grid-kind', provenance[str(args.channels)]['certificate_grid_kind'])
    elif args.action == 'verify':
        # Recheck every continuous cell and the analytic endpoint extensions.
        run(ROOT / 'code/verification/verify_parallel.py', certificate,
            '--verifier', ROOT / 'code/verification/verify_signed_lo_certificate.py',
            '--result', out / f'verified_m{args.channels}.json',
            '--workers', args.workers, '--bands', args.bands)
    elif args.action == 'smoke':
        run(ROOT / 'code/smoke_test.py')


if __name__ == '__main__':
    main()
