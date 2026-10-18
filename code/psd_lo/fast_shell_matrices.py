"""Prefix-sum evaluation of the existing signed-shell certificate bounds.

Discovery arithmetic only. The unchanged exact verifier is authoritative.
Sorted shell radii allow each piecewise polynomial sum over s to be obtained
from prefix sums of w/s, w, and w*s, without a quadratic shell loop.
"""
import numpy as np
from numba import njit, prange


@njit(cache=True)
def prefix(weights,r):
    out=np.zeros((len(r),3,len(r)+1))
    for i in range(len(r)):
        for j in range(len(r)):
            out[i,0,j+1]=out[i,0,j]+weights[i,j]/r[j]
            out[i,1,j+1]=out[i,1,j]+weights[i,j]
            out[i,2,j+1]=out[i,2,j]+weights[i,j]*r[j]
    return out


@njit(cache=True)
def interval_sum(r,P,i,lo,hi,cm,c0,cp,open_left=False):
    if hi<lo:return 0.
    l=np.searchsorted(r,lo,'right') if open_left else np.searchsorted(r,lo,'left')
    h=np.searchsorted(r,hi,'right')
    return cm*(P[i,0,h]-P[i,0,l])+c0*(P[i,1,h]-P[i,1,l])+cp*(P[i,2,h]-P[i,2,l])


@njit(cache=True)
def fast_kernel(a,b,r,P):
    result=0.
    for i in range(len(r)):
        R=r[i]/a
        low=b*abs(R-1);high=b*(R+1)
        l=np.searchsorted(r,low,'right');h=np.searchsorted(r,high,'right')
        if R>=1:result+=(1-1/R)*P[i,1,l]
        result+=(b*(R-1)**2*(P[i,0,h]-P[i,0,l])+2*(R-1)*(P[i,1,h]-P[i,1,l])+(P[i,2,h]-P[i,2,l])/b)/(4*R)
        result+=P[i,1,-1]-P[i,1,h]-b*(P[i,0,-1]-P[i,0,h])
    return result


@njit(cache=True)
def fast_curvature(a,b,c,d,r,pos,neg):
    result=0.
    for i in range(len(r)):
        R=r[i]/a
        hi=d*(R-1)
        j=np.searchsorted(r,hi,'right')
        result+=b*(b-a)**2/(4*r[i]*a*a)*pos[i,1,j]
        lo=max(c*(1-R),c*(r[i]/b-1),0.)
        hi=d*(1+R)
        total=interval_sum(r,neg,i,lo,hi,R*R,0.,0.)
        # Replace the clipped constant on the two regions below the cap.
        L=max(lo,d*(1-R));U=min(hi,c*(1+R))
        mid=2*c*d/(c+d)
        if L<U:
            top=min(U,mid)
            if L<top:
                total+=interval_sum(r,neg,i,L,top,1-R*R,-2/d,1/(d*d))
            bottom=max(L,mid)
            if bottom<U:
                total+=interval_sum(r,neg,i,bottom,U,1-R*R,-2/c,1/(c*c),bottom==mid and L<mid)
        result+=b*d*(b-a)**2/(16*r[i]*a*a)*total
    return result


@njit(parallel=True,cache=True)
def fast_matrices(edges,r,weights):
    n=len(edges)-1
    assert np.all(r[1:]>r[:-1])
    P=prefix(weights,r);pos=prefix(np.maximum(weights,0),r);neg=prefix(np.maximum(-weights,0),r)
    K=np.zeros((n+1,n+1));E=np.zeros((n,n))
    for i in prange(n+1):
        for j in range(i,n+1):
            value=fast_kernel(edges[i],edges[j],r,P)
            K[i,j]=value;K[j,i]=value
        if i<n:
            a,b=edges[i],edges[i+1]
            for j in range(i,n):
                c,d=edges[j],edges[j+1]
                value=fast_curvature(a,b,c,d,r,pos,neg)+fast_curvature(c,d,a,b,r,pos,neg)
                E[i,j]=max(value,0.);E[j,i]=E[i,j]
    return K,E
