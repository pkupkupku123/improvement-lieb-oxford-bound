"""Rational continuous majorant for independently supported channel profiles.

Each channel is quantized on its own <=100 shells. No eigenvector mixing
or rank truncation is performed. Acceptance requires the exact verifier.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,hashlib,json,sys,time,warnings
from pathlib import Path
import numpy as np
from numba import njit,prange,set_num_threads
from scipy.optimize import linprog,OptimizeWarning
from scipy.sparse import coo_matrix
warnings.filterwarnings('ignore',category=OptimizeWarning)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'signed_lo'))
from build_certificate import kernel,curvature,corner_rhs,violations,quantize
sys.path.insert(0,str(ROOT/'psd_lo'))
from build_covariance_certificate import exact_totals


@njit(cache=True,parallel=True)
def sparse_matrices(edges,left,right,weights):
    n=len(edges)-1;K=np.zeros((n+1,n+1));E=np.zeros((n,n))
    for i in prange(n+1):
        for j in range(i,n+1):
            val=0.
            for t in range(len(weights)):val+=weights[t]*kernel(edges[i],edges[j],left[t],right[t])
            K[i,j]=val;K[j,i]=val
        if i<n:
            a,b=edges[i],edges[i+1]
            for j in range(i,n):
                c,d=edges[j],edges[j+1];err=0.
                for t in range(len(weights)):
                    rr,ss,w=left[t],right[t],weights[t]
                    err+=abs(w)*(curvature(a,b,c,d,rr,ss,w>0)+curvature(c,d,a,b,ss,rr,w>0))
                E[i,j]=err;E[j,i]=err
    return K,E


def quantized_measures(profile,cs,ms):
    radii=[[int(round(float(x)*cs)) for x in row] for row in profile['radii']]
    weights=[quantize(profile['weights'][0],ms,True)]
    for row in profile['weights'][1:]:
        z=[int(round(float(x)*ms)) for x in row];z[int(np.argmax(np.abs(row)))]-=sum(z);weights.append(z)
    assert all(len(row)==100 for row in radii+weights)
    assert all(all(a<b for a,b in zip(row,row[1:])) for row in radii)
    assert sum(weights[0])==ms and all(sum(z)==0 for z in weights[1:])
    union=sorted(set(x for row in radii for x in row));index={x:i for i,x in enumerate(union)}
    mapped=[]
    for rr,ww in zip(radii,weights):
        v=[0]*len(union)
        for radius,weight in zip(rr,ww):v[index[radius]]+=weight
        mapped.append(v)
    return union,mapped[0],mapped[1:],dict(radii=radii,weights=weights)


def certificate_edges(intervals,delta,cutoff,cs,kind):
    if kind=='logarithmic':
        nodes=np.geomspace(delta,cutoff,intervals+1)
    elif kind=='focused':
        # Mu's parameter outer radius is normalized to one in this experiment.
        # Concentrate intervals where the shell interaction varies most.
        knots=np.asarray([delta,.015,.05,3.,15.,cutoff])
        if intervals<16 or np.min(np.diff(knots))<=0:
            raise ValueError('Focused grid needs >=16 intervals and delta < .015 < 15 < cutoff')
        counts=np.maximum(1,np.floor(intervals*np.asarray([2,1,10,1,2])/16).astype(int))
        counts[2]+=intervals-int(counts.sum())
        if counts[2]<1:raise ValueError('Too few focused intervals')
        nodes=np.r_[np.concatenate([np.geomspace(a,b,int(n)+1)[:-1]
                                    for a,b,n in zip(knots,knots[1:],counts)]),cutoff]
    else:raise ValueError(kind)
    result=[int(round(float(x)*cs)) for x in nodes]
    assert len(result)==intervals+1 and all(a<b for a,b in zip(result,result[1:]))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--intervals',type=int,default=6400);ap.add_argument('--threads',type=int,default=56)
    ap.add_argument('--delta',type=float,default=1e-4);ap.add_argument('--cutoff',type=float,default=1e5)
    ap.add_argument('--fast-matrices',action='store_true')
    ap.add_argument('--grid-kind',choices=['logarithmic','focused'],default='logarithmic')
    args=ap.parse_args();set_num_threads(args.threads);started=time.monotonic()
    raw=Path(args.input).read_bytes();profile=json.loads(raw);out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    assert profile['format']=='independent-channel-lo-profile-v1' and profile['channels'] in (1,2,3,4,5)
    cs,ms,gs=10**12,10**10,10**18
    ri,pi,zi,measures=quantized_measures(profile,cs,ms)
    r=np.asarray(ri,float)/cs;p=np.asarray(pi,float)/ms;W=np.outer(p,p)
    for row in zi:z=np.asarray(row,float)/ms;W-=np.outer(z,z)
    ii,jj=np.nonzero(W);left=r[ii];right=r[jj];weights=W[ii,jj]
    edge_int=certificate_edges(args.intervals,args.delta,args.cutoff,cs,args.grid_kind)
    edges=np.asarray(edge_int,float)/cs
    print(json.dumps(dict(event='matrices_start',atoms=len(ri),nonzero_terms=len(weights),channels=len(zi),intervals=args.intervals)),flush=True)
    if args.fast_matrices:
        from fast_shell_matrices import fast_matrices
        K,E=fast_matrices(edges,r,W)
    else:
        K,E=sparse_matrices(edges,left,right,weights)
    rhs=corner_rhs(edges,K,E);del K,E
    print(json.dumps(dict(event='matrices_ready',seconds=time.monotonic()-started)),flush=True)
    n=len(edges);dx=1/edges[:-1]-1/edges[1:];floor=2*np.maximum(W,0).sum()*edges[0]**3
    scale=np.maximum(np.minimum(edges**3,1.),floor);cost=.5*(np.r_[dx,0]+np.r_[0,dx])*scale
    bounds=np.column_stack((floor/scale,np.full(n,np.inf)));objective_scale=1e5
    anchors=list(range(0,n,max(1,n//80)))+[n-1]
    active=set(i*n+j for i in range(n) for j in anchors if j>=i);active.update(i*n+i for i in range(n))
    x=bounds[:,0].copy()
    for iteration in range(400):
        indices,values=violations(edges,scale,rhs,x,24)
        for i in range(n):
            for j,v in zip(indices[i],values[i]):
                if j>=0 and v>2e-10:
                    active.add(i*n+int(j))
                    if iteration>=4:
                        for nearby in range(max(i,int(j)-8),min(n,int(j)+9)):active.add(i*n+nearby)
        keys=np.asarray(sorted(active),dtype=np.int64);a,b=keys//n,keys%n
        ca,cb=scale[a]/edges[b]**3,scale[b]/edges[a]**3;norm=np.maximum(ca,cb);ca/=norm;cb/=norm
        rows=np.arange(len(keys));matrix=coo_matrix((-np.r_[ca,cb],(np.r_[rows,rows],np.r_[a,b])),shape=(len(keys),n)).tocsc()
        options={'threads':1,'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8}
        result=linprog(cost*objective_scale,A_ub=matrix,b_ub=-rhs[a,b]/norm,bounds=bounds,method='highs',options=options)
        if not result.success:result=linprog(cost*objective_scale,A_ub=matrix,b_ub=-rhs[a,b]/norm,bounds=bounds,method='highs-ds',options=dict(options,presolve=False))
        if not result.success:raise RuntimeError(result.message)
        x=result.x;_,residual=violations(edges,scale,rhs,x,24)
        print(json.dumps(dict(event='cutting_plane',iteration=iteration,constraints=len(keys),max_violation=float(residual.max()),seconds=time.monotonic()-started)),flush=True)
        if residual.max()<2e-8:break
    else:raise RuntimeError('Continuous LP cutting planes did not converge')
    h=(x*(1+3e-7)+2e-7)*scale
    certificate=dict(format='covariance-lo-linear-reciprocal-v1',coordinate_scale=cs,mass_scale=ms,allocation_scale=gs,
                     radii=ri,mu=pi,nu=pi,covariance_channels=zi,edges=edge_int,
                     allocation=[int(np.ceil(v*gs)) for v in h],claimed_constant='9')
    totals=exact_totals(certificate);certificate['claimed_constant']=totals['proposed_constant_decimal']
    out.write_text(json.dumps(certificate,indent=2))
    mapping=out.with_suffix('.measures.json');mapping.write_text(json.dumps(measures,indent=2))
    report=dict(status='PROPOSED_NOT_VERIFIED',input_sha256=hashlib.sha256(raw).hexdigest(),
                certificate_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),quantized_measures_sha256=hashlib.sha256(mapping.read_bytes()).hexdigest(),
                channels=len(zi),shells_per_measure=100,union_atoms=len(ri),intervals=args.intervals,
                slurm_job_id=os.environ.get('SLURM_JOB_ID'),fast_matrices=args.fast_matrices,
                grid_kind=args.grid_kind,seconds=time.monotonic()-started,**totals)
    out.with_suffix('.proposed.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_exact')},indent=2),flush=True)


if __name__=='__main__':main()
