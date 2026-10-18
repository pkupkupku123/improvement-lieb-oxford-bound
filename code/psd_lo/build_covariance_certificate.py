"""Build a rational full-domain LO certificate with extra covariance channels.

Reuses the established shell curvature and interpolation bounds unchanged.
The output is only proposed; verify_signed_lo_certificate.py must accept it.
The input is a saved PSD discovery output, not a mutable optimization run.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import hashlib
import json
import sys
import time
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
from numba import set_num_threads
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'signed_lo'))
from build_certificate import matrices, corner_rhs, violations, quantize


def channel_integers(discovery, mass_scale, threshold=1e-13):
    vectors=np.asarray(discovery['column_vectors'])
    coeff=np.asarray(discovery['column_coefficients'])
    if len(vectors)==0:
        return []
    covariance=(vectors.T*coeff)@vectors
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    channels=[]
    for value,vector in zip(eigenvalues,eigenvectors.T):
        if value>threshold:
            z=np.sqrt(value)*vector
            z-=z.mean()
            zi=[int(round(float(x)*mass_scale)) for x in z]
            zi[int(np.argmax(np.abs(z))) ]-=sum(zi)
            assert sum(zi)==0
            channels.append(zi)
    return channels


def exact_totals(d):
    cs,ms,gs=(d[k] for k in ('coordinate_scale','mass_scale','allocation_scale'))
    r,p,q=(d[k] for k in ('radii','mu','nu')); z=d['covariance_channels']
    w=[[p[i]*q[j]+q[i]*p[j]-q[i]*q[j]-sum(v[i]*v[j] for v in z)
        for j in range(len(r))] for i in range(len(r))]
    assert sum(map(sum,w))==ms*ms
    L,G=d['edges'],d['allocation']; P=sum(max(t,0) for row in w for t in row)
    origin=Q(P*L[0]**2,2*ms*ms*cs*cs)
    tail=Q(27*sum(max(w[i][j],0)*r[j]**3 for i in range(len(r)) for j in range(len(r))),128*ms*ms*cs*cs*L[-1])
    middle=sum((Q(-((-v.numerator*10**18)//v.denominator),10**18)
                for v in (Q((gi+gj)*cs*(b-a),2*gs*a*b) for a,b,gi,gj in zip(L,L[1:],G,G[1:]))),Q(0))
    energy=sum((Q(p[i]*p[j]*cs,2*ms*ms*max(r[i],r[j])) for i in range(len(r)) for j in range(len(r))),Q(0))
    cube=Q(27,2)*Q(355,113)*energy**2*(origin+middle+tail)
    dec=10**9; claim=int(float(cube)**(1/3)*dec)
    while Q(claim,dec)**3<cube:claim+=1
    while Q(claim-1,dec)**3>=cube:claim-=1
    return dict(origin_exact=str(origin),tail_exact=str(tail),middle_upper_exact=str(middle),
                energy_exact=str(energy),cube_upper_exact=str(cube),
                proposed_constant_decimal=f'{claim/dec:.9f}',constant_approx=float(cube)**(1/3))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--intervals',type=int,default=6400);ap.add_argument('--threads',type=int,default=6)
    ap.add_argument('--delta',type=float,default=.0001);ap.add_argument('--cutoff',type=float,default=100000.)
    ap.add_argument('--without-correction',action='store_true')
    ap.add_argument('--fast-matrices',action='store_true')
    args=ap.parse_args();set_num_threads(args.threads)
    start=time.monotonic();raw=Path(args.input).read_bytes();profile=json.loads(raw)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    cs,ms,gs=10**12,10**10,10**18
    ri=[int(round(x*cs)) for x in profile['radii']]
    pi=quantize(profile['mu'],ms,True);qi=quantize(profile['nu'],ms)
    zi=[] if args.without_correction else channel_integers(profile,ms)
    r=np.asarray(ri,dtype=float)/cs;p=np.asarray(pi,dtype=float)/ms;q=np.asarray(qi,dtype=float)/ms
    W=np.outer(p,q)+np.outer(q,p)-np.outer(q,q)
    for zz in zi:
        v=np.asarray(zz,dtype=float)/ms;W-=np.outer(v,v)
    edge_int=[int(round(z*cs)) for z in np.geomspace(args.delta,args.cutoff,args.intervals+1)]
    edges=np.asarray(edge_int,dtype=float)/cs
    print(json.dumps(dict(event='building_matrices',shells=len(r),channels=len(zi),intervals=args.intervals)),flush=True)
    if args.fast_matrices:
        from fast_shell_matrices import fast_matrices
        K,E=fast_matrices(edges,r,W)
    else:
        K,E=matrices(edges,r,W)
    print(json.dumps(dict(event='matrices_ready',seconds=time.monotonic()-start)),flush=True)
    rhs=corner_rhs(edges,K,E);del K,E
    n=len(edges);dx=1/edges[:-1]-1/edges[1:]
    floor=2*np.maximum(W,0).sum()*edges[0]**3
    scale=np.maximum(np.minimum(edges**3,1),floor)
    cost=.5*(np.r_[dx,0]+np.r_[0,dx])*scale
    objective_scale=100000.
    bounds=np.column_stack((floor/scale,np.full(n,np.inf)))
    anchors=list(range(0,n,max(1,n//80)))+[n-1]
    active=set(i*n+j for i in range(n) for j in anchors if j>=i)
    active.update(i*n+i for i in range(n));x=bounds[:,0].copy()
    for iteration in range(400):
        idx,val=violations(edges,scale,rhs,x,24)
        worst=float(val.max())
        for i in range(n):
            for j,v in zip(idx[i],val[i]):
                if j>=0 and v>2e-10:
                    active.add(i*n+int(j))
                    if iteration>=4:
                        for nearby in range(max(i,int(j)-8),min(n,int(j)+9)):
                            active.add(i*n+nearby)
        keys=np.asarray(sorted(active),dtype=np.int64);ii,jj=keys//n,keys%n
        ca,cb=scale[ii]/edges[jj]**3,scale[jj]/edges[ii]**3
        norm=np.maximum(ca,cb);ca/=norm;cb/=norm
        rows=np.arange(len(keys))
        matrix=coo_matrix((-np.r_[ca,cb],(np.r_[rows,rows],np.r_[ii,jj])),shape=(len(keys),n)).tocsr()
        result=linprog(cost*objective_scale,A_ub=matrix,b_ub=-rhs[ii,jj]/norm,bounds=bounds,method='highs',
                       options={'threads':1,'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
        if not result.success:
            result=linprog(cost*objective_scale,A_ub=matrix,b_ub=-rhs[ii,jj]/norm,bounds=bounds,method='highs-ipm',
                           options={'threads':1,'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
        if not result.success:raise RuntimeError(result.message)
        x=result.x
        print(json.dumps(dict(event='cutting_plane',iteration=iteration,constraints=len(keys),
                              previous_max_violation=worst,integral_middle=float(result.fun/objective_scale),
                              seconds=time.monotonic()-start)),flush=True)
        _,val=violations(edges,scale,rhs,x,24)
        if val.max()<2e-8:break
    else:raise RuntimeError('Cutting planes did not converge')
    h=(x*(1+3e-7)+2e-7)*scale
    data=dict(format='covariance-lo-linear-reciprocal-v1',coordinate_scale=cs,mass_scale=ms,
              allocation_scale=gs,radii=ri,mu=pi,nu=qi,covariance_channels=zi,
              edges=edge_int,allocation=[int(np.ceil(z*gs)) for z in h],claimed_constant='9')
    totals=exact_totals(data);data['claimed_constant']=totals['proposed_constant_decimal']
    out.write_text(json.dumps(data,indent=2))
    report=dict(status='PROPOSED_NOT_YET_VERIFIED',input=args.input,input_sha256=hashlib.sha256(raw).hexdigest(),
                certificate_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),channels=len(zi),
                intervals=args.intervals,fast_matrices=args.fast_matrices,seconds=time.monotonic()-start,**totals)
    out.with_suffix('.proposed.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_exact')}),flush=True)


if __name__=='__main__':main()
