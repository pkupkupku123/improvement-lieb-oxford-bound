"""Full-domain parallel execution of the original exact signed-LO verifier.

Only its initial rectangle is changed. Contiguous first-coordinate bands
partition [0,n] and each covers the entire second coordinate [0,n]. The
original symmetry skip, subdivision, curvature checks, endpoint bounds,
and final rational cube comparison are unmodified. Every worker checks
the same certificate and recomputes its global arithmetic.
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor,as_completed
import contextlib
import hashlib
import json
from pathlib import Path
import time


def worker(task):
    source,certificate,lo,hi,log=task
    tree=ast.parse(Path(source).read_text())
    found=[]
    expected=ast.dump(ast.parse('pending = [(0, n, 0, n)]').body[0].value)
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='pending':
            if ast.dump(node.value)!=expected:raise RuntimeError('Unexpected original initial rectangle')
            node.value=ast.parse(f'pending = [({lo}, {hi}, 0, n)]').body[0].value
            found.append(node)
    if len(found)!=1:raise RuntimeError('Expected exactly one initial rectangle')
    ast.fix_missing_locations(tree)
    namespace={'__name__':'original_signed_verifier_shard','__file__':str(source)}
    with open(log,'w') as stream,contextlib.redirect_stdout(stream):
        exec(compile(tree,str(source),'exec'),namespace)
        result=namespace['verify'](certificate)
    result['first_coordinate_band']=[lo,hi]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('certificate');p.add_argument('--verifier',required=True)
    p.add_argument('--result',required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--bands',type=int,default=64)
    a=p.parse_args();start=time.monotonic();certificate=Path(a.certificate).resolve();source=Path(a.verifier).resolve()
    original_source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    if original_source_hash!='c9108a741ad99484bbef7c03fc217bde2894bf415c4a22d9b3e6c027023ebb04':
        raise RuntimeError('Original verifier differs from the previously audited version')
    raw=certificate.read_bytes();digest=hashlib.sha256(raw).hexdigest();data=json.loads(raw)
    n=len(data['edges'])-1;count=min(a.bands,n)
    cuts=[n*i//count for i in range(count+1)]
    assert cuts[0]==0 and cuts[-1]==n and all(x<y for x,y in zip(cuts,cuts[1:]))
    result_path=Path(a.result);result_path.parent.mkdir(parents=True,exist_ok=True)
    logs=result_path.with_suffix('.shards');logs.mkdir(exist_ok=True)
    tasks=[(str(source),str(certificate),lo,hi,str(logs/f'band_{lo:05d}_{hi:05d}.log')) for lo,hi in zip(cuts,cuts[1:])]
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(worker,t) for t in tasks]
        for f in as_completed(futures):
            r=f.result()
            assert r['status']=='PASS' and r['certificate_sha256']==digest and r['intervals']==n
            results.append(r)
            if len(results)%8==0 or len(results)==count:
                print(json.dumps(dict(event='bands_passed',completed=len(results),total=count,seconds=time.monotonic()-start)),flush=True)
    results.sort(key=lambda x:x['first_coordinate_band'])
    assert [x['first_coordinate_band'] for x in results]==[[x,y] for x,y in zip(cuts,cuts[1:])]
    fields=['claimed_constant_exact','shells','intervals','self_energy_approx','integral_upper_approx',
            'origin_integral_approx','tail_integral_approx','constant_upper_approx','nu_total_variation_approx',
            'additional_covariance_channels']
    for field in fields:assert all(r[field]==results[0][field] for r in results)
    record=dict(status='PASS',verification_scope='Entire continuous domain, including analytic origin and tail',
                certificate_sha256=digest,original_verifier_sha256=original_source_hash,
                parallel_verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                workers=a.workers,bands=count,seconds=time.monotonic()-start,
                coverage='Contiguous disjoint first-coordinate bands; full second coordinate; original symmetric skip',
                **{key:results[0][key] for key in fields},
                **{key:sum(r[key] for r in results) for key in ['rectangles_checked','monotone_blocks','curvature_cells']},
                shard_results=results)
    assert hashlib.sha256(certificate.read_bytes()).hexdigest()==digest
    result_path.write_text(json.dumps(record,indent=2))
    print(json.dumps({k:v for k,v in record.items() if k!='shard_results'},indent=2),flush=True)


if __name__=='__main__':main()
