"""Discrete variational HV metric used as the signed-signal baseline.

The implementation follows the published HV dynamic functional and eliminates
the auxiliary horizontal field by solving the reduced SPD system at each
interpolation time level. It is a discrete variational reimplementation, not
the older fixed-point MATLAB port.

Reference: R. Han, D. Slepcev, and Y. Yang, "HV geometry for signal
comparison", Quarterly of Applied Mathematics 82 (2024), 391--430.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

def matrices(nx):
    dx=1.0/(nx-1);w=np.ones(nx);w[[0,-1]]=0.5
    D1=np.zeros((nx,nx))
    for i in range(1,nx-1):D1[i,i-1]=-1/(2*dx);D1[i,i+1]=1/(2*dx)
    D1[0,0]=-1/dx;D1[0,1]=1/dx;D1[-1,-2]=-1/dx;D1[-1,-1]=1/dx
    D2=np.zeros((nx,nx))
    for i in range(1,nx-1):D2[i,i-1]=1/dx**2;D2[i,i]=-2/dx**2;D2[i,i+1]=1/dx**2
    B=np.zeros((nx,nx-2));B[1:-1]=np.eye(nx-2);W=np.diag(w)
    return dx,w,D1,D2,B,W

def solve_hv(f0,f1,kappa=1e-4,lam=1e-4,eps=1e-12,nt=16,maxiter=300,gtol=1e-8,x0=None):
    f0=np.asarray(f0,float);f1=np.asarray(f1,float);nx=f0.size;dx,w,D1,D2,B,W=matrices(nx);dt=1/(nt-1)
    Rfull=kappa*W+lam*(D1.T@W@D1)+eps*(D2.T@W@D2);R=B.T@Rfull@B;wi=w[1:-1]
    if x0 is None:
        a=np.linspace(0,1,nt)[:,None];F0=(1-a)*f0[None]+a*f1[None];x0=F0[1:-1].ravel()
    def valgrad(x, return_aux=False):
        F=np.empty((nt,nx));F[0]=f0;F[-1]=f1;F[1:-1]=x.reshape(nt-2,nx)
        Ft=(F[1:]-F[:-1])/dt;Fx=F[:-1]@D1.T
        A=np.broadcast_to(R,(nt-1,nx-2,nx-2)).copy()
        ii=np.arange(nx-2);A[:,ii,ii]+=wi[None,:]*Fx[:,1:-1]**2
        rhs=-(wi[None,:]*Fx[:,1:-1]*Ft[:,1:-1])
        u=np.linalg.solve(A,rhs[...,None])[...,0];V=u@B.T;Z=Ft+V*Fx
        vreg=np.einsum('bi,ij,bj->b',u,R,u);zterm=np.sum(w[None,:]*Z*Z,axis=1)
        J=.5*dx*dt*np.sum(vreg+zterm)
        G=np.zeros_like(F)
        wz=w[None,:]*Z
        G[:-1]+= -dx*wz + dx*dt*((V*wz)@D1)
        G[1:]+= dx*wz
        g=G[1:-1].ravel()
        if return_aux:return J,g,G,V,Z,F
        return J,g
    res=minimize(valgrad,np.asarray(x0,float),jac=True,method='L-BFGS-B',options={'maxiter':maxiter,'ftol':1e-13,'gtol':gtol,'maxls':40,'maxcor':15})
    J,g,G,V,Z,F=valgrad(res.x,True)
    return {'action':float(J),'grad_f0':G[0].copy(),'x':res.x,'success':bool(res.success),'nit':int(res.nit),'nfev':int(res.nfev),'message':str(res.message),'opt_grad_inf':float(np.max(np.abs(g))),'V':V,'Z':Z,'F':F}
