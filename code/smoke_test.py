"""Small real execution tests, including rejection of an invalid certificate.

The synthetic one-shell example is deliberately loose. It exercises the exact
verifier and multiprocessing dispatcher without pretending to prove paper data.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / 'code/verification/verify_signed_lo_certificate.py'


def main():
    if not __debug__:
        raise RuntimeError('Run with assertions enabled')
    spec = importlib.util.spec_from_file_location('exact', VERIFIER)
    exact = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exact)
    assert exact.shell_fraction(2, 2, 1, 1) == (0, 1)
    assert exact.shell_fraction(1, 1, 1, 1) == (1, 4)
    assert exact.shell_fraction(1, 1, 3, 1) == (2, 3)
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        candidate = dict(format='covariance-lo-linear-reciprocal-v1', coordinate_scale=1,
                         mass_scale=1, allocation_scale=1, radii=[1], mu=[1], nu=[1],
                         covariance_channels=[[0]], edges=[1, 2, 4], allocation=[100, 100, 100],
                         claimed_constant='20')
        path = folder / 'example.json'
        path.write_text(json.dumps(candidate))
        direct = exact.verify(path)
        subprocess.run([sys.executable, '-B', str(ROOT / 'code/verification/verify_parallel.py'),
                        str(path), '--verifier', str(VERIFIER), '--result', str(folder / 'parallel.json'),
                        '--workers', '2', '--bands', '2'], check=True)
        parallel = json.loads((folder / 'parallel.json').read_text())
        assert parallel['constant_upper_approx'] == direct['constant_upper_approx']
        candidate['claimed_constant'] = '1'
        path.write_text(json.dumps(candidate))
        try:
            exact.verify(path)
        except AssertionError:
            pass
        else:
            raise RuntimeError('Invalid constant was accepted')
    print('PASS: shell branches, serial/parallel full verification of a small example, invalid-claim rejection.')


if __name__ == '__main__':
    main()
