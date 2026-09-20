import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def lift(y, eps=0.15, omega=None):
    y=np.asarray(y,float).ravel(); n=y.size
    if omega is None: omega=np.ones(n)
    omega=np.asarray(omega,float).ravel(); Mw=omega.sum()
    s2=(omega*y*y).sum()/Mw
    r=np.sqrt(y*y + eps*eps*s2)
    Z=(omega*r).sum()
    qp=0.5*omega*(r+y); qm=0.5*omega*(r-y)
    rho=np.r_[qp,qm]/Z
    return rho, (r,Z,s2,omega)


def jt_eta(y, eta, eps=0.15, omega=None):
    y=np.asarray(y,float).ravel(); n=y.size
    if omega is None: omega=np.ones(n)
    rho,(r,Z,s2,omega)=lift(y,eps,omega)
    ep=np.asarray(eta[:n]); em=np.asarray(eta[n:])
    cp=ep@rho[:n] + em@rho[n:]
    wt=omega*(ep+em-2*cp)/(2*Z)
    beta=eps*eps/omega.sum()
    term1=omega*(ep-em)/(2*Z)
    term2=wt*(y/r)
    term3=beta*(omega*y)*((1/r)@wt)
    return term1+term2+term3


def jacobian_dense(y, eps=0.15, omega=None):
    y=np.asarray(y,float).ravel(); n=y.size
    if omega is None: omega=np.ones(n)
    omega=np.asarray(omega,float).ravel(); Mw=omega.sum(); W=np.diag(omega)
    rho,(r,Z,s2,omega)=lift(y,eps,omega)
    beta=eps*eps/Mw
    Dr=np.diag(y/r) + beta*np.outer(1/r, omega*y)
    qp=0.5*omega*(r+y); qm=0.5*omega*(r-y)
    Dqp=0.5*W@(Dr+np.eye(n)); Dqm=0.5*W@(Dr-np.eye(n))
    dZ=omega@Dr
    J=np.vstack([Dqp,Dqm])/Z - np.outer(np.r_[qp,qm],dZ)/(Z*Z)
    return J


def two_layer_graph_edges(shape, cross_weight=0.25, between_block_weight=0.2):
    """Undirected nearest-neighbor edges on a Cartesian acquisition graph, duplicated for two species.
    For multidimensional shapes, axis 0 gets between_block_weight (e.g. source/pattern coupling),
    remaining axes get unit weight. Same-site cross-species edges ensure connectedness.
    """
    if isinstance(shape,int): shape=(shape,)
    shape=tuple(int(v) for v in shape); N=int(np.prod(shape)); base=[]
    for idx in np.ndindex(*shape):
        i=np.ravel_multi_index(idx,shape)
        for ax in range(len(shape)):
            if idx[ax]+1 < shape[ax]:
                jdx=list(idx); jdx[ax]+=1; jdx=tuple(jdx)
                j=np.ravel_multi_index(jdx,shape)
                q=between_block_weight if (len(shape)>1 and ax==0) else 1.0
                base.append((i,j,q))
    edges=[]
    for off in (0,N):
        edges += [(i+off,j+off,q) for i,j,q in base]
    edges += [(i,i+N,cross_weight) for i in range(N)]
    return edges


def B_from_rho(rho, edges):
    rho=np.asarray(rho,float); M=rho.size
    rows=[]; cols=[]; vals=[]; diag=np.zeros(M)
    for i,j,q in edges:
        w=q*0.5*(rho[i]+rho[j])
        diag[i]+=w; diag[j]+=w
        rows += [i,j]; cols += [j,i]; vals += [-w,-w]
    rows += list(range(M)); cols += list(range(M)); vals += list(diag)
    return sp.csr_matrix((vals,(rows,cols)),shape=(M,M))


def solve_laplacian_pinv(B, b):
    """Pseudoinverse action for connected graph Laplacian on zero-sum b.
    Fix last node gauge, solve reduced system, then center. Energy invariant to centering.
    """
    b=np.asarray(b,float).ravel()
    b=b-b.mean()  # guard roundoff
    Br=B[:-1,:-1].tocsc(); xr=spla.spsolve(Br,b[:-1])
    x=np.r_[xr,0.0]; x-=x.mean(); return x


def frozen_setup(d, eps=0.15, data_shape=None, cross_weight=0.25, between_block_weight=0.2):
    d=np.asarray(d,float).ravel(); rho,_=lift(d,eps)
    if data_shape is None: data_shape=(d.size,)
    edges=two_layer_graph_edges(data_shape,cross_weight,between_block_weight)
    B=B_from_rho(rho,edges)
    solver=spla.factorized(B[:-1,:-1].tocsc())
    return {'d':d,'rho_d':rho,'B':B,'edges':edges,'eps':eps,'shape':data_shape,'solver':solver}


def frozen_objective_grad(y, setup):
    y=np.asarray(y,float).ravel(); rho,_=lift(y,setup['eps'])
    dr=rho-setup['rho_d']; b=dr-dr.mean(); xr=setup['solver'](b[:-1]); eta=np.r_[xr,0.0]; eta-=eta.mean()
    val=0.5*float(dr@eta)
    gy=jt_eta(y,eta,setup['eps'])
    return val,gy


def normalized_l2_objective_grad(y,d):
    y=np.asarray(y,float).ravel(); d=np.asarray(d,float).ravel()
    ny=np.linalg.norm(y); nd=np.linalg.norm(d)
    uy=y/ny; ud=d/nd; r=uy-ud
    val=0.5*r@r
    gy=(r - uy*(uy@r))/ny
    return float(val),gy


def l2_objective_grad(y,d):
    r=np.asarray(y,float).ravel()-np.asarray(d,float).ravel()
    scale=max(np.linalg.norm(d)**2,1e-30)
    return 0.5*float(r@r)/scale, r/scale



def fixed_scale_lift(y, reference_scale2, eps=0.15, omega=None):
    y=np.asarray(y,float).ravel(); n=y.size
    if omega is None: omega=np.ones(n)
    omega=np.asarray(omega,float).ravel()
    r=np.sqrt(y*y + eps*eps*float(reference_scale2))
    Z=(omega*r).sum()
    qp=0.5*omega*(r+y); qm=0.5*omega*(r-y)
    rho=np.r_[qp,qm]/Z
    return rho, (r,Z,omega)


def fixed_scale_jt_eta(y, eta, reference_scale2, eps=0.15, omega=None):
    y=np.asarray(y,float).ravel(); n=y.size
    if omega is None: omega=np.ones(n)
    omega=np.asarray(omega,float).ravel()
    rho,(r,Z,omega)=fixed_scale_lift(y,reference_scale2,eps,omega)
    ep=np.asarray(eta[:n]); em=np.asarray(eta[n:])
    cp=ep@rho[:n] + em@rho[n:]
    wt=omega*(ep+em-2*cp)/(2*Z)
    term1=omega*(ep-em)/(2*Z)
    term2=wt*(y/r)
    return term1+term2


def fixed_scale_setup(d, eps=0.15, data_shape=None, cross_weight=0.25, between_block_weight=0.2):
    d=np.asarray(d,float).ravel()
    reference_scale2=float(np.mean(d*d))
    rho,_=fixed_scale_lift(d,reference_scale2,eps)
    if data_shape is None: data_shape=(d.size,)
    edges=two_layer_graph_edges(data_shape,cross_weight,between_block_weight)
    B=B_from_rho(rho,edges)
    solver=spla.factorized(B[:-1,:-1].tocsc())
    return {'d':d,'rho_d':rho,'B':B,'edges':edges,'eps':eps,'shape':data_shape,
            'solver':solver,'reference_scale2':reference_scale2}


def fixed_scale_objective_grad(y, setup):
    y=np.asarray(y,float).ravel()
    rho,_=fixed_scale_lift(y,setup['reference_scale2'],setup['eps'])
    dr=rho-setup['rho_d']; b=dr-dr.mean()
    xr=setup['solver'](b[:-1]); eta=np.r_[xr,0.0]; eta-=eta.mean()
    val=0.5*float(dr@eta)
    gy=fixed_scale_jt_eta(y,eta,setup['reference_scale2'],setup['eps'])
    return val,gy
