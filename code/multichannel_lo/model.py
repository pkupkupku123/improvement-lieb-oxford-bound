"""Independent radial supports for mu and each zero-mass Coulomb channel.

Numerical discovery only. The signed shell matrix is blockwise ww^T for mu
and -ww^T for each channel; no cross terms between channels are introduced.
Prefix sums evaluate each block in O(shells log shells), and LP gradients
are contracted only over nonzero dual multipliers.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import warnings
import numpy as np
from numba import njit,prange
from scipy.optimize import linprog,OptimizeWarning
from scipy.sparse import coo_matrix
warnings.filterwarnings('ignore',category=OptimizeWarning)


@njit(cache=True)
def prefixes(r,w):
    d,m=r.shape;out=np.zeros((d,3,m+1))
    for c in range(d):
        for j in range(m):
            out[c,0,j+1]=out[c,0,j]+w[c,j]/r[c,j]
            out[c,1,j+1]=out[c,1,j]+w[c,j]
            out[c,2,j+1]=out[c,2,j]+w[c,j]*r[c,j]
    return out


@njit(cache=True)
def row_sums(a,b,ri,r,P):
    """Sum_j w_j k(ri/a,rj/b), and its log(ri) derivative."""
    R=ri/a
    low=b*abs(R-1.);high=b*(R+1.)
    l=np.searchsorted(r,low,'right');h=np.searchsorted(r,high,'right')
    v0=P[1,h]-P[1,l];vm=P[0,h]-P[0,l];vp=P[2,h]-P[2,l]
    value=(b*(R-1.)**2*vm+2*(R-1.)*v0+vp/b)/(4*R)
    dr=(b*(R*R-1.)*vm+2*v0-vp/b)/(4*R)
    if R>=1.:
        value+=(1-1/R)*P[1,l];dr+=P[1,l]/R
    value+=P[1,-1]-P[1,h]-b*(P[0,-1]-P[0,h])
    return value,dr


@njit(cache=True,parallel=True)
def kernel_values(a,b,r,w,P):
    out=np.zeros(len(a));d,m=r.shape
    for t in prange(len(a)):
        value=0.
        for c in range(d):
            if r[c,-1]/a[t]+r[c,-1]/b[t]<=1.:continue
            subtotal=0.
            for i in range(m):
                v,_=row_sums(a[t],b[t],r[c,i],r[c],P[c])
                subtotal+=w[c,i]*v
            value+=subtotal if c==0 else -subtotal
        out[t]=value
    return out


@njit(cache=True,parallel=True)
def contracted_gradient(a,b,multiplier,r,w,P):
    d,m=r.shape;dw=np.zeros((d,m));dlr=np.zeros((d,m))
    for feature in prange(d*m):
        c=feature//m;i=feature%m;gw=gr=0.
        for t in range(len(a)):
            if r[c,-1]/a[t]+r[c,-1]/b[t]<=1.:continue
            va,ra=row_sums(a[t],b[t],r[c,i],r[c],P[c])
            vb,rb=row_sums(b[t],a[t],r[c,i],r[c],P[c])
            gw+=multiplier[t]*(va+vb)
            gr+=multiplier[t]*w[c,i]*(ra+rb)
        sign=1. if c==0 else -1.
        dw[c,i]=sign*gw;dlr[c,i]=sign*gr
    return dw,dlr


def endpoint_terms(r,w,delta,cutoff):
    """Separate block positive parts: valid also when different blocks overlap."""
    dw=np.zeros_like(w);dr=np.zeros_like(r);dp=np.zeros_like(w);P=E=0.
    for c in range(len(r)):
        sign=1. if c==0 else -1.
        W=sign*np.outer(w[c],w[c]);positive=np.maximum(W,0)
        B=delta**2/2+27/(128*cutoff)*r[c][None,:]**3
        P+=positive.sum();E+=float(np.sum(positive*B))
        G=(W>0)*B;G=(G+G.T)/2
        dw[c]=2*sign*G@w[c]
        dr[c]=81/(128*cutoff)*r[c]**3*positive.sum(axis=0)
        dp[c]=2*sign*(W>0)@w[c]
    return float(P),float(E),dw,dr,dp


class Model:
    def __init__(self,channels,shells=100,intervals=880,delta=1e-4,cutoff=1e5,nodes=None):
        self.channels=channels;self.d=channels+1;self.m=shells
        self.delta=delta;self.cutoff=cutoff;self.intervals=intervals
        self.nodes=np.geomspace(delta,cutoff,intervals+1) if nodes is None else np.asarray(nodes)
        self.ii,self.jj=np.triu_indices(len(self.nodes));self.a=self.nodes[self.ii];self.b=self.nodes[self.jj]
        self.scale=np.minimum(self.nodes**3,1.)
        ca=self.scale[self.ii]/self.b**3;cb=self.scale[self.jj]/self.a**3
        self.norm=np.maximum(ca,cb);ca/=self.norm;cb/=self.norm
        rows=np.arange(len(ca))
        self.matrix=coo_matrix((-np.r_[ca,cb],(np.r_[rows,rows],np.r_[self.ii,self.jj])),
                              shape=(len(rows),len(self.nodes))).tocsc()
        dx=1/self.nodes[:-1]-1/self.nodes[1:]
        self.cost=.5*(np.r_[dx,0]+np.r_[0,dx])*self.scale
        self.objective_scale=1e5;self.lp_retries=0

    def pack(self,r,w):
        return np.r_[w.ravel(),np.delete(np.log(r).ravel(),self.m-1)]

    def unpack(self,x):
        n=self.d*self.m
        return np.exp(np.insert(x[n:],self.m-1,0.)).reshape(self.d,self.m),x[:n].reshape(self.d,self.m)

    def solve_lp(self,rhs,bounds):
        options={'threads':1,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9}
        lp=linprog(self.cost*self.objective_scale,A_ub=self.matrix,b_ub=rhs,bounds=bounds,
                   method='highs',options=options)
        if not lp.success:
            self.lp_retries+=1
            lp=linprog(self.cost*self.objective_scale,A_ub=self.matrix,b_ub=rhs,bounds=bounds,
                       method='highs-ds',options=dict(options,presolve=False))
        if not lp.success:raise RuntimeError(lp.message)
        self.last_lp_info=dict(lp_method='all_rows',lp_separation_iterations=0,lp_rows=len(rhs))
        return lp

    def evaluate(self,x,gradient=True,allocation=False):
        r,w=self.unpack(x)
        # SLSQP may evaluate infeasible intermediate radius orders. Sorting
        # preserves the represented measures; gradients are permuted back.
        order=np.argsort(r,axis=1);r=np.take_along_axis(r,order,axis=1);w=np.take_along_axis(w,order,axis=1)
        Pfx=prefixes(r,w)
        K=kernel_values(self.a,self.b,r,w,Pfx)
        P,E,ew,er,pw=endpoint_terms(r,w,self.delta,self.cutoff)
        floor=2*P*self.delta**3
        lower=floor/self.scale;bounds=np.column_stack((lower,np.full(len(lower),np.inf)))
        rhs=-2*K/self.norm
        lp=self.solve_lp(rhs,bounds)
        A=float(self.cost@lp.x+E);V=1/np.maximum.outer(r[0],r[0]);p=w[0];e=float(.5*p@V@p)
        C=float((27*np.pi/2*e*e*A)**(1/3))
        details=dict(constant=C,energy=e,integral=A,endpoint=E,positive_mass_bound=P,
                     lp_violation=float(max(0,np.max(self.matrix@lp.x-rhs))),
                     lower_bound_violation=float(max(0,np.max(lower-lp.x))),
                     active_rows=int(np.count_nonzero(lp.ineqlin.marginals)),
                     channel_total_variations=np.abs(w[1:]).sum(axis=1).tolist(),**self.last_lp_info)
        if allocation:details.update(nodes=self.nodes.tolist(),allocation=(lp.x*self.scale).tolist())
        if not gradient:return C,details
        dual=-lp.ineqlin.marginals/self.objective_scale;keep=dual>0
        dw,dr=contracted_gradient(self.a[keep],self.b[keep],2*dual[keep]/self.norm[keep],r,w,Pfx)
        floor_sensitivity=float((lp.lower.marginals/self.objective_scale)@(2*self.delta**3/self.scale))
        dw+=ew+floor_sensitivity*pw;dr+=er
        de_w=np.zeros_like(w);de_r=np.zeros_like(r);de_w[0]=V@p
        de_r[0]=-(p*np.r_[0.,np.cumsum(p)[:-1]]+.5*p*p)/r[0]
        gw=C/3*(dw/A+2*de_w/e);gr=C/3*(dr/A+2*de_r/e)
        original_gw=np.empty_like(gw);original_gr=np.empty_like(gr)
        np.put_along_axis(original_gw,order,gw,axis=1);np.put_along_axis(original_gr,order,gr,axis=1)
        gw,gr=original_gw,original_gr
        return C,np.r_[gw.ravel(),np.delete(gr.ravel(),self.m-1)],details

    def constraints(self,min_radius=1e-4,max_channel_radius=4.,min_log_gap=1e-4,weight_limit=16.):
        n=self.d*self.m;dim=2*n-1
        masses=np.zeros((self.d,dim));target=np.zeros(self.d);target[0]=1
        order=np.zeros((self.d*(self.m-1),dim))
        full_to_var={j:n+j-(j>self.m-1) for j in range(n) if j!=self.m-1}
        for c in range(self.d):
            masses[c,c*self.m:(c+1)*self.m]=1
            for i in range(self.m-1):
                row=c*(self.m-1)+i
                for index,sign in ((c*self.m+i,-1),(c*self.m+i+1,1)):
                    if index in full_to_var:order[row,full_to_var[index]]=sign
        cs=[dict(type='eq',fun=lambda x:masses@x-target,jac=lambda x:masses),
            dict(type='ineq',fun=lambda x:order@x-min_log_gap,jac=lambda x:order)]
        bounds=[(0.,1.)]*self.m+[(-weight_limit,weight_limit)]*(self.channels*self.m)
        bounds += [(np.log(min_radius),0. if j<self.m else np.log(max_channel_radius))
                   for j in range(n) if j!=self.m-1]
        return cs,bounds


def load_profile(path):
    import json
    from pathlib import Path
    data=json.loads(Path(path).read_bytes())
    assert data['format']=='independent-channel-lo-profile-v1'
    return np.asarray(data['radii'],float),np.asarray(data['weights'],float),data
