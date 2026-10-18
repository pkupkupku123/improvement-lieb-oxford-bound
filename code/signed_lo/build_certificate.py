"""Propose a rational continuum certificate. Acceptance is a separate exact check."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import json
import time
from pathlib import Path
from fractions import Fraction as Q
import numpy as np
from numba import njit,prange,set_num_threads
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from search_profile import load_profile


@njit
def kernel(u,v,r,s):
    rb,sa,uv=r*v,s*u,u*v
    if rb+sa<=uv:return 0.
    if rb-sa>=uv:return 1.-u/r
    if sa-rb>=uv:return 1.-v/s
    return (rb+sa-uv)**2/(4*r*s*u*v)


@njit
def curvature(a,b,c,d,r,s,positive):
    if positive:
        if r*d-s*a<a*d:return 0.
        return b*(b-a)**2/(4*r*a*a)
    if r*c+s*a<a*c or r*c-s*b>b*c or s*a-r*d>a*d:return 0.
    t=min(max((s/d-1)**2,(s/c-1)**2),(r/a)**2)
    return b*d*(b-a)**2*t/(16*r*s*a*a)


@njit(parallel=True)
def matrices(edges,radii,weights):
    n=len(edges)-1;k=len(radii)
    K=np.zeros((n+1,n+1)); E=np.zeros((n,n))
    for i in prange(n+1):
        for j in range(i,n+1):
            total=0.
            for r in range(k):
                for s in range(k):
                    w=weights[r,s]
                    if w!=0:total+=w*kernel(edges[i],edges[j],radii[r],radii[s])
            K[i,j]=total;K[j,i]=total
        if i<n:
            a,b=edges[i],edges[i+1]
            for j in range(i,n):
                c,d=edges[j],edges[j+1];error=0.
                for r in range(k):
                    for s in range(k):
                        w=weights[r,s]
                        if w!=0:
                            error+=abs(w)*(curvature(a,b,c,d,radii[r],radii[s],w>0)+
                                           curvature(c,d,a,b,radii[s],radii[r],w>0))
                E[i,j]=error;E[j,i]=error
    return K,E


@njit(parallel=True)
def corner_rhs(edges,K,E):
    n=len(edges)-1;rhs=np.zeros((n+1,n+1))
    eps=(edges[1:]-edges[:-1])**2*(edges[:-1]+2*edges[1:])/(4*edges[:-1]**3)
    for i in prange(n+1):
        for j in range(i,n+1):
            value=0.
            for a in range(max(0,i-1),min(i+1,n)):
                for b in range(max(0,j-1),min(j+1,n)):
                    value=max(value,2*(1+max(eps[a],eps[b]))*(K[i,j]+E[a,b]))
            rhs[i,j]=value;rhs[j,i]=value
    return rhs


@njit(parallel=True)
def violations(edges,scale,rhs,x,count):
    n=len(edges); indices=np.full((n,count),-1,np.int64); values=np.zeros((n,count))
    for i in prange(n):
        for j in range(i,n):
            ca=scale[i]/edges[j]**3;cb=scale[j]/edges[i]**3
            d=max(ca,cb)
            v=(rhs[i,j]-ca*x[i]-cb*x[j])/d
            bucket=min(count-1,(j-i)*count//(n-i))
            if v>values[i,bucket]:values[i,bucket]=v;indices[i,bucket]=j
    return indices,values


def quantize(values,scale,positive=False):
    nums=[int(round(float(x)*scale)) for x in values]
    if positive:nums=[max(x,0) for x in nums]
    nums[int(np.argmax(values))]+=scale-sum(nums)
    assert sum(nums)==scale and (not positive or min(nums)>=0)
    return nums


def rational_totals(data):
    cs,ms,gs=(data[k] for k in ('coordinate_scale','mass_scale','allocation_scale'))
    r,p,q=(data[k] for k in ('radii','mu','nu'));L,G=data['edges'],data['allocation']
    w=[[p[i]*q[j]+q[i]*p[j]-q[i]*q[j] for j in range(len(r))] for i in range(len(r))]
    P=sum(max(a,0) for row in w for a in row)
    origin=Q(P*L[0]**2,2*ms*ms*cs*cs)
    tail=Q(27*sum(max(w[i][j],0)*r[j]**3 for i in range(len(r)) for j in range(len(r))),128*ms*ms*cs*cs*L[-1])
    middle=sum((Q((gi+gj)*cs*(b-a),2*gs*a*b) for a,b,gi,gj in zip(L,L[1:],G,G[1:])),Q(0))
    # Match the original verifier's per-interval upward rounding.
    middle_up=sum((Q(-((-z.numerator*10**18)//z.denominator),10**18)
                   for z in (Q((gi+gj)*cs*(b-a),2*gs*a*b) for a,b,gi,gj in zip(L,L[1:],G,G[1:]))),Q(0))
    energy=sum((Q(p[i]*p[j]*cs,2*ms*ms*max(r[i],r[j])) for i in range(len(r)) for j in range(len(r))),Q(0))
    cube=Q(27,4)*energy**2*2*Q(355,113)*(origin+middle_up+tail)
    dec=10**9;claim=int(round(float(cube)**(1/3)*dec))
    while Q(claim,dec)**3<cube:claim+=1
    while Q(claim-1,dec)**3>=cube:claim-=1
    return dict(self_energy_exact=str(energy),origin_exact=str(origin),tail_exact=str(tail),
                middle_exact=str(middle),middle_upper_exact=str(middle_up),cube_upper_exact=str(cube),
                proposed_constant_exact=str(Q(claim,dec)),proposed_constant_decimal=f'{claim/dec:.9f}',
                constant_approx=float(cube)**(1/3))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--intervals',type=int,default=3200);ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--delta',type=float,default=.0001);ap.add_argument('--cutoff',type=float,default=100000)
    args=ap.parse_args();set_num_threads(args.threads);out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();r,p,q=load_profile(args.input);cs=10**12;ms=10**10;gs=10**18
    ri=[int(round(x*cs)) for x in r];pi=quantize(p,ms,True);qi=quantize(q,ms)
    r=np.asarray(ri,dtype=float)/cs;p=np.asarray(pi,dtype=float)/ms;q=np.asarray(qi,dtype=float)/ms
    edge_int=[int(round(z*cs)) for z in np.geomspace(args.delta,args.cutoff,args.intervals+1)]
    assert all(a<b for a,b in zip(edge_int,edge_int[1:]));edges=np.asarray(edge_int,dtype=float)/cs
    w=np.outer(p,q)+np.outer(q,p)-np.outer(q,q)
    K,E=matrices(edges,r,w)
    print(json.dumps(dict(event='matrices_ready',seconds=time.monotonic()-start,intervals=args.intervals)),flush=True)
    rhs=corner_rhs(edges,K,E);del K,E
    n=len(edges);dx=1/edges[:-1]-1/edges[1:]
    floor=2*np.maximum(w,0).sum()*edges[0]**3
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
                    # Neighboring cuts prevent a nearly flat active contact
                    # from walking one grid point per LP solve.
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
            print(json.dumps(dict(event='retry_interior_point',message=result.message)),flush=True)
            result=linprog(cost*objective_scale,A_ub=matrix,b_ub=-rhs[ii,jj]/norm,bounds=bounds,method='highs-ipm',
                           options={'threads':1,'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
        if not result.success:raise RuntimeError(result.message)
        x=result.x
        print(json.dumps(dict(event='cutting_plane',iteration=iteration,constraints=len(keys),
                              previous_max_violation=worst,integral_middle=result.fun/objective_scale,seconds=time.monotonic()-start)),flush=True)
        idx,val=violations(edges,scale,rhs,x,24)
        if val.max()<2e-8:break
    else:raise RuntimeError('Cutting planes did not converge')
    # Discovery tolerances never decide validity. Independent exact verification follows.
    h=(x*(1+3e-7)+2e-7)*scale
    G=[int(np.ceil(z*gs)) for z in h]
    data=dict(format='signed-lo-linear-reciprocal-v1',coordinate_scale=cs,mass_scale=ms,
              allocation_scale=gs,radii=ri,mu=pi,nu=qi,edges=edge_int,allocation=G,claimed_constant='9')
    totals=rational_totals(data);data['claimed_constant']=totals['proposed_constant_decimal']
    out.write_text(json.dumps(data,indent=2))
    report=dict(status='PROPOSED_NOT_YET_VERIFIED',input=args.input,intervals=args.intervals,
                shells=len(r),seconds=time.monotonic()-start,**totals)
    out.with_suffix('.proposed.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_exact')}),flush=True)


if __name__=='__main__':main()
