"""Solve the same full-grid LP by constraint generation and complete separation.

Omitted constraints are checked after every solve. The accepted primal is
feasible on the full grid to a stricter tolerance than the original solver;
the dual is extended by zero, so no untested reduced-grid model is used.
"""
import numpy as np
from numba import njit
from scipy.optimize import linprog
from model import Model


@njit(cache=True)
def separating_rows(residual,n,count=8):
    indices=np.full((n,count),-1,np.int64);values=np.zeros((n,count));t=0
    for i in range(n):
        for j in range(i,n):
            bucket=min(count-1,(j-i)*count//(n-i));v=residual[t]
            if v>values[i,bucket]:values[i,bucket]=v;indices[i,bucket]=t
            t+=1
    return indices.ravel(),values.ravel()


class ActiveModel(Model):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.csr=self.matrix.tocsr();self.diagonal=np.flatnonzero(self.ii==self.jj)
        self.active=self.diagonal.copy()

    def neighbors(self,indices):
        n=len(self.nodes);i=self.ii[indices];j=self.jj[indices];rows=[indices,self.diagonal]
        for offset in (-2,-1,1,2):
            jj=np.clip(j+offset,i,n-1);rows.append(i*n-i*(i+1)//2+jj)
        return np.unique(np.concatenate(rows))

    def solve_lp(self,rhs,bounds):
        options={'threads':1,'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10}
        active=self.active.copy();n=len(self.nodes)
        for iteration in range(100):
            lp=linprog(self.cost*self.objective_scale,A_ub=self.csr[active].tocsc(),b_ub=rhs[active],
                       bounds=bounds,method='highs',options=options)
            if not lp.success:
                self.lp_retries+=1
                lp=linprog(self.cost*self.objective_scale,A_ub=self.csr[active].tocsc(),b_ub=rhs[active],
                           bounds=bounds,method='highs-ds',options=dict(options,presolve=False))
            if not lp.success:break
            residual=self.csr@lp.x-rhs
            if max(residual)<=2e-10:
                nonzero=lp.ineqlin.marginals!=0
                self.active=self.neighbors(active[nonzero])
                dual=np.zeros(len(rhs));dual[active]=lp.ineqlin.marginals
                lp.ineqlin.marginals=dual
                self.last_lp_info=dict(lp_method='constraint_generation_full_grid_scan',
                                       lp_separation_iterations=iteration+1,lp_rows=len(active),
                                       lp_complete_grid_max_residual=float(max(0,max(residual))))
                return lp
            rows,values=separating_rows(residual,n);added=rows[(rows>=0)&(values>2e-10)]
            expanded=np.union1d(active,self.neighbors(added))
            if len(expanded)==len(active):break
            active=expanded
        # A numerical difficulty falls back to the original complete LP.
        lp=super().solve_lp(rhs,bounds)
        self.active=self.neighbors(np.flatnonzero(lp.ineqlin.marginals))
        self.last_lp_info['lp_method']='full_grid_fallback'
        return lp
