"""Re-solve the grid LP for one frozen floating-point optimized profile."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / 'multichannel_lo'))
from active_model import ActiveModel
from model import load_profile
from numba import set_num_threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--intervals', type=int, default=3520)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    set_num_threads(args.threads)
    started = time.monotonic()
    r, w, source = load_profile(args.input)
    print(f'Re-solving grid LP: m={len(r)-1}, intervals={args.intervals}, threads={args.threads}', flush=True)
    # Model.pack fixes the outer parameter radius of mu to one.
    r = r / r[0, -1]
    model = ActiveModel(len(r)-1, r.shape[1], args.intervals)
    constant, details = model.evaluate(model.pack(r, w), gradient=False)
    record = dict(status='NUMERICAL_ONLY', channels=len(r)-1, intervals=args.intervals,
                  numerical_objective=constant, archived_objective=source['validation_constant'],
                  seconds=time.monotonic()-started, details=details)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
