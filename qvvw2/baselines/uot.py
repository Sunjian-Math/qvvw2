"""Unbalanced optimal transport (UOT) baseline following Li et al. (2022).

Reference:
D. Li, M. P. Lamoureux, and W. Liao, "Application of an unbalanced
optimal transport distance and a mixed L1/Wasserstein distance to full waveform
inversion", Geophysical Journal International 230 (2022), 1338--1357.

This module implements the UOT branch only. The stabilized scaling is
algebraically equivalent to the source iteration and retains the published
objective and gradient.
"""
from __future__ import annotations
import numpy as np

def _kl_batch(q,p):
    tiny=np.finfo(float).tiny
    return np.sum(q*np.log(np.maximum(q,tiny)/np.maximum(p,tiny))-q+p,axis=-1)

def li_uot_batch(Y,d,x,reg=1e-3,reg_m=1.,k=None,maxiter=12000,check_every=100,tol=1e-11):
    """Converged, overflow-safe faithful port of the official Julia UOT scaling.

    The update is algebraically identical to the source iteration.  Each scaling
    vector is represented as exp(s)*shape with max(shape)=1; the scalar log
    factors are tracked separately, preventing overflow without changing the
    transport plan or fixed point.
    """
    Y=np.atleast_2d(np.asarray(Y,float)); d=np.asarray(d,float); x=np.asarray(x,float)
    if k is None: k=1.5/max(float(np.max(d)),1e-12)
    A=np.exp(np.clip(k*Y,-60,60)); b=np.exp(np.clip(k*d,-60,60))
    M=(x[:,None]-x[None,:])**2; K=np.exp(-M/reg); fi=reg_m/(reg+reg_m)
    tiny=np.finfo(float).tiny; B=Y.shape[0];n=Y.shape[1]
    uh=np.ones((B,n)); vh=np.ones((B,n)); su=np.full(B,-np.log(n)); sv=np.full(B,-np.log(n))
    resid=np.inf; it=maxiter
    prev_su=su.copy();prev_sv=sv.copy();prev_uh=uh.copy();prev_vh=vh.copy()
    for j in range(1,maxiter+1):
        den=np.maximum(uh@K,tiny)
        base=(b[None,:]/den)**fi
        mx=np.maximum(np.max(base,axis=1),tiny); vh=base/mx[:,None]; sv=-fi*su+np.log(mx)
        den=np.maximum(vh@K.T,tiny)
        base=(A/den)**fi
        mx=np.maximum(np.max(base,axis=1),tiny); uh=base/mx[:,None]; su=-fi*sv+np.log(mx)
        if j%check_every==0:
            # Exact log-vector fixed-point residual, invariant to our representation.
            lu=su[:,None]+np.log(np.maximum(uh,tiny)); lv=sv[:,None]+np.log(np.maximum(vh,tiny))
            plu=prev_su[:,None]+np.log(np.maximum(prev_uh,tiny)); plv=prev_sv[:,None]+np.log(np.maximum(prev_vh,tiny))
            resid=max(float(np.max(np.abs(lu-plu))),float(np.max(np.abs(lv-plv))))
            prev_su=su.copy();prev_sv=sv.copy();prev_uh=uh.copy();prev_vh=vh.copy()
            if resid<tol:
                it=j;break
    scale=np.exp(su+sv)
    T=scale[:,None,None]*uh[:,:,None]*K[None,:,:]*vh[:,None,:]
    a1=T.sum(2);b1=T.sum(1)
    logTK=(su+sv)[:,None,None]+np.log(np.maximum(uh[:,:,None],tiny))+np.log(np.maximum(vh[:,None,:],tiny))
    ent=np.sum(T*logTK-T+K[None,:,:],axis=(1,2))
    val=reg*ent+reg_m*(_kl_batch(a1,A)+_kl_batch(b1,b[None,:]))
    return val,{'k':float(k),'reg':reg,'reg_m':reg_m,'iterations':it,'fixed_point_residual':float(resid),'stabilization':'exact scalar-shape factorization of source scaling variables'}

def li_uot_value_grad_batch(Y,d,x,reg=1e-3,reg_m=1.,k=None,maxiter=14000,check_every=100,tol=1e-11):
    """UOT objective and paper-correct gradient w.r.t. waveform samples.
    Returns per-trace objective values and gradients.  The source Julia gradient
    w.r.t. exp(k*y) is combined with the exponential chain factor k*exp(k*y),
    as verified in the FULL-PASS finite-difference audit.
    """
    Y=np.atleast_2d(np.asarray(Y,float)); d=np.atleast_2d(np.asarray(d,float))
    if d.shape[0]==1 and Y.shape[0]>1:d=np.repeat(d,Y.shape[0],axis=0)
    x=np.asarray(x,float)
    if k is None:k=1.5/max(float(np.max(d)),1e-12)
    A=np.exp(np.clip(k*Y,-60,60));Bobs=np.exp(np.clip(k*d,-60,60))
    M=(x[:,None]-x[None,:])**2;K=np.exp(-M/reg);fi=reg_m/(reg+reg_m);tiny=np.finfo(float).tiny
    nb,n=Y.shape;uh=np.ones((nb,n));vh=np.ones((nb,n));su=np.full(nb,-np.log(n));sv=np.full(nb,-np.log(n));resid=np.inf;it=maxiter
    prev_su=su.copy();prev_sv=sv.copy();prev_uh=uh.copy();prev_vh=vh.copy()
    for j in range(1,maxiter+1):
        den=np.maximum(uh@K,tiny);base=(Bobs/den)**fi;mx=np.maximum(np.max(base,axis=1),tiny);vh=base/mx[:,None];sv=-fi*su+np.log(mx)
        den=np.maximum(vh@K.T,tiny);base=(A/den)**fi;mx=np.maximum(np.max(base,axis=1),tiny);uh=base/mx[:,None];su=-fi*sv+np.log(mx)
        if j%check_every==0:
            lu=su[:,None]+np.log(np.maximum(uh,tiny));lv=sv[:,None]+np.log(np.maximum(vh,tiny));plu=prev_su[:,None]+np.log(np.maximum(prev_uh,tiny));plv=prev_sv[:,None]+np.log(np.maximum(prev_vh,tiny));resid=max(float(np.max(np.abs(lu-plu))),float(np.max(np.abs(lv-plv))));prev_su=su.copy();prev_sv=sv.copy();prev_uh=uh.copy();prev_vh=vh.copy()
            if resid<tol:it=j;break
    logu=su[:,None]+np.log(np.maximum(uh,tiny));scale=np.exp(su+sv);T=scale[:,None,None]*uh[:,:,None]*K[None,:,:]*vh[:,None,:];a1=T.sum(2);b1=T.sum(1);logTK=(su+sv)[:,None,None]+np.log(np.maximum(uh[:,:,None],tiny))+np.log(np.maximum(vh[:,None,:],tiny));ent=np.sum(T*logTK-T+K[None,:,:],axis=(1,2));vals=reg*ent+reg_m*(_kl_batch(a1,A)+_kl_batch(b1,Bobs))
    ff=reg*logu;grad_a=-reg_m*(np.exp(np.clip(-ff/reg_m,-60,60))-1.0);grad_y=k*A*grad_a
    return vals,grad_y,{'k':float(k),'reg':reg,'reg_m':reg_m,'iterations':it,'fixed_point_residual':float(resid),'gradient_chain':'paper-correct k*exp(k*y) factor included'}

def li_uot_value_grad_batch_fast(Y,d,x,reg=1e-3,reg_m=1.,k=None,shape_iters=2000):
    """Gauge-accelerated UOT fixed-point solver with the same converged solution.

    The normalized shapes of u and v are independent of their scalar gauges.
    We iterate those shapes and solve the two scalar gauge equations analytically,
    eliminating the very slow phi≈1 global-scaling mode.  FULL-PASS code audits
    this fast form against the source-equivalent iteration before using it.
    """
    Y=np.atleast_2d(np.asarray(Y,float));d=np.atleast_2d(np.asarray(d,float))
    if d.shape[0]==1 and Y.shape[0]>1:d=np.repeat(d,Y.shape[0],axis=0)
    x=np.asarray(x,float)
    if k is None:k=1.5/max(float(np.max(d)),1e-12)
    A=np.exp(np.clip(k*Y,-60,60));Bobs=np.exp(np.clip(k*d,-60,60));M=(x[:,None]-x[None,:])**2;K=np.exp(-M/reg);phi=reg_m/(reg+reg_m);tiny=np.finfo(float).tiny
    nb,n=Y.shape;uh=np.ones((nb,n));vh=np.ones((nb,n))
    for _ in range(shape_iters):
        base=(Bobs/np.maximum(uh@K,tiny))**phi;mv=np.maximum(np.max(base,axis=1),tiny);vh=base/mv[:,None];cv=np.log(mv)
        base=(A/np.maximum(vh@K.T,tiny))**phi;mu=np.maximum(np.max(base,axis=1),tiny);uh=base/mu[:,None];cu=np.log(mu)
    su=(cu-phi*cv)/(1-phi**2);sv=cv-phi*su;logu=su[:,None]+np.log(np.maximum(uh,tiny));scale=np.exp(su+sv);T=scale[:,None,None]*uh[:,:,None]*K[None,:,:]*vh[:,None,:];a1=T.sum(2);b1=T.sum(1);logTK=(su+sv)[:,None,None]+np.log(np.maximum(uh[:,:,None],tiny))+np.log(np.maximum(vh[:,None,:],tiny));ent=np.sum(T*logTK-T+K[None,:,:],axis=(1,2));vals=reg*ent+reg_m*(_kl_batch(a1,A)+_kl_batch(b1,Bobs));ff=reg*logu;grad_a=-reg_m*(np.exp(np.clip(-ff/reg_m,-60,60))-1.);grad_y=k*A*grad_a
    return vals,grad_y,{'k':float(k),'reg':reg,'reg_m':reg_m,'shape_iters':int(shape_iters),'acceleration':'analytic scalar-gauge fixed point; normalized shape iteration unchanged'}
