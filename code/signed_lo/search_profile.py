"""Numerical discovery only: signed radial LO profiles and an inner LP.

No output of this program is a certified bound. The separate rational
verifier must accept a continuum certificate before a bound is claimed.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix


def shell_kernel(u, v, r, s):
    R, S = r/u, s/v
    out = np.maximum(R+S-1, 0)**2/(4*R*S)
    out = np.where(R-S >= 1, 1-1/R, out)
    out = np.where(S-R >= 1, 1-1/S, out)
    return np.where(R+S <= 1, 0, out)


def load_profile(path):
    d = json.loads(Path(path).read_text())
    r, p, q = (np.asarray(d[k], dtype=float) for k in ('radii','mu','nu'))
    if 'coordinate_scale' in d:
        r /= d['coordinate_scale']; p /= d['mass_scale']; q /= d['mass_scale']
    return r, p, q


def expand_profile(r, p, q, count):
    original = r.copy()
    points = list(r)
    while len(points) < count:
        points.sort()
        gaps = np.log(np.asarray(points[1:])/points[:-1])
        i = int(np.argmax(gaps))
        points.append(float(np.sqrt(points[i]*points[i+1])))
    points = np.asarray(sorted(points))
    pp = np.zeros(len(points)); qq = pp.copy()
    for rr, a, b in zip(original,p,q):
        i = int(np.argmin(abs(points-rr))); pp[i] = a; qq[i] = b
    return points, pp, qq


class InnerLP:
    def __init__(self, radii, intervals, delta=0.0002, cutoff=10000.0):
        self.r = np.asarray(radii); self.k = len(radii)
        self.u = np.geomspace(delta,cutoff,intervals+1)
        self.delta, self.cutoff = delta, cutoff
        self.ii,self.jj = np.triu_indices(len(self.u))
        a,b = self.u[self.ii],self.u[self.jj]
        self.scale = np.minimum(self.u**3,1.)
        ca,cb = self.scale[self.ii]/b**3,self.scale[self.jj]/a**3
        self.rowscale = np.maximum(ca,cb)
        ca /= self.rowscale; cb /= self.rowscale
        rows = np.arange(len(a))
        self.matrix = coo_matrix((-np.r_[ca,cb],(np.r_[rows,rows],np.r_[self.ii,self.jj])),
                                 shape=(len(a),len(self.u))).tocsr()
        dx = 1/self.u[:-1]-1/self.u[1:]
        self.cost = .5*(np.r_[dx,0]+np.r_[0,dx])*self.scale
        self.kernel = np.empty((len(a),self.k*self.k),dtype=np.float64)
        started = time.monotonic()
        for i,r in enumerate(self.r):
            for j,s in enumerate(self.r):
                self.kernel[:,i*self.k+j] = shell_kernel(a,b,r,s)
        self.kernel = np.ascontiguousarray(self.kernel)
        self.V = 1/np.maximum.outer(self.r,self.r)
        print(json.dumps(dict(event='kernel_ready',shells=self.k,nodes=len(self.u),
                              megabytes=self.kernel.nbytes/1e6,seconds=time.monotonic()-started)),flush=True)
        self.calls = 0

    def evaluate(self,p,q,gradient=True):
        w = np.outer(p,q)+np.outer(q,p)-np.outer(q,q)
        kernel = self.kernel @ w.ravel()
        rhs = 2*kernel/self.rowscale
        result = linprog(self.cost,A_ub=self.matrix,b_ub=-rhs,bounds=(0,None),
                         method='highs', options={'threads':1,'dual_feasibility_tolerance':1e-9,
                                                 'primal_feasibility_tolerance':1e-9})
        if not result.success:
            raise RuntimeError(result.message)
        positive = np.maximum(w,0)
        endpoint_matrix = self.delta**2/2 + 27/128/self.cutoff*self.r[None,:]**3
        endpoint = np.sum(positive*endpoint_matrix)
        A = result.fun+endpoint
        e = .5*p@self.V@p
        C = (27*np.pi/2*e*e*A)**(1/3)
        if not gradient:
            return dict(constant=C,energy=e,integral=A,endpoint=endpoint,
                        h=result.x*self.scale,nodes=self.u,lp=result)
        dual = -result.ineqlin.marginals
        gradW = (self.kernel.T @ (2*dual/self.rowscale)).reshape(self.k,self.k)
        gradW += (w>0)*endpoint_matrix
        gradW = .5*(gradW+gradW.T)
        gradp = C/3*(2*(self.V@p)/e + 2*gradW@q/A)
        gradq = C/3*(2*gradW@(p-q)/A)
        self.calls += 1
        return C,np.r_[gradp,gradq]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',required=True); ap.add_argument('--output',required=True)
    ap.add_argument('--shells',type=int,default=50); ap.add_argument('--intervals',type=int,default=220)
    ap.add_argument('--maxiter',type=int,default=100); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--gradient-check',action='store_true'); ap.add_argument('--evaluate-only',action='store_true')
    args=ap.parse_args(); out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    r,p,q=load_profile(args.input)
    if args.shells>len(r):r,p,q=expand_profile(r,p,q,args.shells)
    if args.seed:
        rng=np.random.default_rng(args.seed)
        p=.99*p+.01*rng.dirichlet(np.ones(len(p)))
        z=rng.normal(0,.02,len(q)); q=q+z-z.mean()
    lp=InnerLP(r,args.intervals); initial=np.r_[p,q]; best=[np.inf,None]; history=[]
    started=time.monotonic()
    def objective(x):
        c,g=lp.evaluate(x[:len(r)],x[len(r):])
        if c<best[0] and abs(x[:len(r)].sum()-1)<1e-7 and abs(x[len(r):].sum()-1)<1e-7:
            best[:]=[float(c),x.copy()]
            record=dict(format='signed-lo-profile-v1',status='NUMERICAL_ONLY',radii=r.tolist(),
                        mu=x[:len(r)].tolist(),nu=x[len(r):].tolist(),
                        numerical_constant=float(c),intervals=args.intervals,seed=args.seed,
                        calls=lp.calls,seconds=time.monotonic()-started)
            out.write_text(json.dumps(record,indent=2))
            history.append({k:record[k] for k in ('numerical_constant','calls','seconds')})
            print(json.dumps(dict(event='best',**history[-1])),flush=True)
        return c,g
    c0,g0=objective(initial)
    if args.gradient_check:
        rng=np.random.default_rng(271828)
        for index in range(4):
            d=rng.normal(size=2*len(r)); d[:len(r)]-=d[:len(r)].mean(); d[len(r):]-=d[len(r):].mean()
            if index<2:
                d[:len(r)]=0
            else:
                d[len(r):]=0
                active=initial[:len(r)]>1e-4
                d[:len(r)][~active]=0
                d[:len(r)][active]-=d[:len(r)][active].mean()
            d/=np.linalg.norm(d)
            h=1e-5
            a=lp.evaluate((initial+h*d)[:len(r)],(initial+h*d)[len(r):])[0]
            b=lp.evaluate((initial-h*d)[:len(r)],(initial-h*d)[len(r):])[0]
            print(json.dumps(dict(event='gradient_check',analytic=float(g0@d),finite_difference=(a-b)/(2*h))),flush=True)
    if not args.evaluate_only:
        n=len(r)
        constraints={'type':'eq','fun':lambda x:np.array([x[:n].sum()-1,x[n:].sum()-1]),
                     'jac':lambda x:np.array([np.r_[np.ones(n),np.zeros(n)],np.r_[np.zeros(n),np.ones(n)]])}
        result=minimize(objective,initial,jac=True,method='SLSQP',bounds=[(0,1)]*n+[(-8,8)]*n,
                        constraints=constraints,options={'maxiter':args.maxiter,'ftol':2e-9,'disp':True})
        print(json.dumps(dict(event='search_finished',success=bool(result.success),message=str(result.message),
                              best=best[0],iterations=result.nit,calls=lp.calls)),flush=True)
    out.with_suffix('.history.json').write_text(json.dumps(history,indent=2))


if __name__=='__main__':main()
