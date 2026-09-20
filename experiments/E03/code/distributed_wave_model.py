from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
import os,math
import numpy as np, torch
import torch.nn.functional as F
import scipy.sparse.linalg as spla
from qvvw2.core import lift,B_from_rho,jt_eta,l2_objective_grad,normalized_l2_objective_grad

torch.set_num_threads(4); DTYPE=torch.float64
P=8;n_inv=49;n_obs=65;ns=4;nrec=20
Tnp=np.linspace(.28,1.18,72);T=torch.tensor(Tnp,dtype=DTYPE)

def ricker(t,f=7,t0=.14):
 a=(math.pi*f*(t-t0))**2; return (1-2*a)*torch.exp(-a)
def pos_ring(n,k,offset=0):
 dx=1/(n-1); th=2*np.pi*(np.arange(k)+offset)/k; r=.39; return [(int(round((.5+r*np.cos(a))/dx)),int(round((.5+r*np.sin(a))/dx))) for a in th]
def sim(c,f=7,tmax=1.25):
 n=c.shape[0];dx=1/(n-1);dt=.42*dx/(math.sqrt(2)*1.18);nt=int(tmax/dt)+1
 src=pos_ring(n,ns,.25);rec=pos_ring(n,nrec,0);u0=torch.zeros((ns,n,n),dtype=DTYPE);u1=torch.zeros_like(u0);out=[]
 sp=max(5,int(.1/dx));damp=torch.ones((n,n),dtype=DTYPE)
 for i in range(sp):
  q=(sp-i)/sp;fac=math.exp(-.12*q*q);damp[i,:]*=fac;damp[-1-i,:]*=fac;damp[:,i]*=fac;damp[:,-1-i]*=fac
 ts=torch.arange(nt,dtype=DTYPE)*dt;sv=ricker(ts,f);coef=(dt*c/dx)**2
 for k in range(nt):
  lap=u1[:,2:,1:-1]+u1[:,:-2,1:-1]+u1[:,1:-1,2:]+u1[:,1:-1,:-2]-4*u1[:,1:-1,1:-1]
  un=torch.zeros_like(u1);un[:,1:-1,1:-1]=2*u1[:,1:-1,1:-1]-u0[:,1:-1,1:-1]+coef[1:-1,1:-1]*lap
  for s,(i,j) in enumerate(src):un[s,i,j]+=0.025*sv[k]
  un*=damp;out.append(torch.stack([un[:,i,j] for i,j in rec],1));u0,u1=u1,un
 return dt,torch.stack(out,-1)
def rs(y,dt):
 q=T/dt;i=torch.floor(q).long().clamp(0,y.shape[-1]-2);a=q-i.to(q.dtype);return y[...,i]*(1-a)+y[...,i+1]*a

def truec(n):
 x=torch.linspace(0,1,n,dtype=DTYPE);X,Z=torch.meshgrid(x,x,indexing='ij');g1=torch.exp(-((X-.36)**2+(Z-.48)**2)/(2*.095**2));g2=torch.exp(-((X-.66)**2+(Z-.61)**2)/(2*.105**2));return 1+.105*g1-.085*g2

def cfrom(m,n=n_inv):
 mi=F.interpolate(m[None,None],size=(n,n),mode='bicubic',align_corners=True)[0,0]
 x=torch.linspace(0,1,n,dtype=DTYPE);X,Z=torch.meshgrid(x,x,indexing='ij');tap=torch.sin(math.pi*X)**2*torch.sin(math.pi*Z)**2
 return 1+.14*torch.tanh(mi)*tap
with torch.no_grad():dto,yo=sim(truec(n_obs));d=rs(yo,dto).numpy();ctr=truec(n_inv).numpy()

def edges_ring(shape,source_w=.03,rec_w=.08,time_w=1.,cross=.08):
 sh=shape;N=int(np.prod(sh));base=[]
 for idx in np.ndindex(*sh):
  i=np.ravel_multi_index(idx,sh)
  # source neighbor no wrap? use ring wrap
  for ax,w,wrap in [(0,source_w,True),(1,rec_w,True),(2,time_w,False)]:
   if idx[ax]+1<sh[ax]:
    jj=list(idx);jj[ax]+=1;j=np.ravel_multi_index(tuple(jj),sh);base.append((i,j,w))
   elif wrap and idx[ax]==sh[ax]-1:
    jj=list(idx);jj[ax]=0;j=np.ravel_multi_index(tuple(jj),sh);base.append((i,j,w))
 edges=[]
 for off in (0,N):edges += [(i+off,j+off,w) for i,j,w in base]
 edges += [(i,i+N,cross) for i in range(N)]
 return edges

def qsetup(d,eps=.15,**kw):
 rho,_=lift(d.ravel(),eps);B=B_from_rho(rho,edges_ring(d.shape,**kw));sol=spla.factorized(B[:-1,:-1].tocsc());return rho,sol,eps

def qog(y,st):
 rd,sol,eps=st;rho,_=lift(y.ravel(),eps);dr=rho-rd;b=dr-dr.mean();xr=sol(b[:-1]);eta=np.r_[xr,0.];eta-=eta.mean();return .5*float(dr@eta),jt_eta(y.ravel(),eta,eps)

def metrics(c):
 mask=np.ones_like(c,bool);mask[:5]=mask[-5:]=False;mask[:,:5]=False;mask[:,-5:]=False
 return np.linalg.norm((c-ctr)[mask])/np.linalg.norm((ctr-1)[mask]),np.corrcoef((c[mask]-1).ravel(),(ctr[mask]-1).ravel())[0,1]

