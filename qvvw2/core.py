import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from qvvw2.input_validation import finite_array, positive_weights, acquisition_shape, graph_edges


def lift(y, eps=0.15, omega=None):
    y=finite_array(y,name='signal').ravel(); n=y.size
    if not np.any(y!=0): raise ValueError('scale-invariant lift requires a nonzero signal.')
    if not np.isfinite(eps) or eps<=0: raise ValueError('eps must be finite and positive.')
    omega=positive_weights(omega,n); Mw=omega.sum()
    s2=(omega*y*y).sum()/Mw
    r=np.sqrt(y*y + eps*eps*s2)
    Z=(omega*r).sum()
    # Rationalization avoids cancellation in the smaller species when eps is small.
    large=r+np.abs(y)
    small=(eps*eps*s2)/large
    qp=0.5*omega*np.where(y>=0,large,small)
    qm=0.5*omega*np.where(y>=0,small,large)
    rho=np.r_[qp,qm]/Z
    return rho, (r,Z,s2,omega)


def jt_eta(y, eta, eps=0.15, omega=None):
    y=finite_array(y,name='signal').ravel(); n=y.size
    eta=finite_array(eta,name='eta').ravel()
    if eta.size != 2*n: raise ValueError('eta must have exactly twice the signal length.')
    rho,(r,Z,s2,omega)=lift(y,eps,omega)
    ep=eta[:n]; em=eta[n:]
    cp=ep@rho[:n] + em@rho[n:]
    wt=omega*(ep+em-2*cp)/(2*Z)
    beta=eps*eps/omega.sum()
    term1=omega*(ep-em)/(2*Z)
    term2=wt*(y/r)
    term3=beta*(omega*y)*((1/r)@wt)
    return term1+term2+term3


def jacobian_dense(y, eps=0.15, omega=None):
    y=finite_array(y,name='signal').ravel(); n=y.size
    omega=positive_weights(omega,n); Mw=omega.sum(); W=np.diag(omega)
    rho,(r,Z,s2,omega)=lift(y,eps,omega)
    beta=eps*eps/Mw
    Dr=np.diag(y/r) + beta*np.outer(1/r, omega*y)
    # Rationalization avoids cancellation in the smaller species when eps is small.
    large=r+np.abs(y)
    small=(eps*eps*s2)/large
    qp=0.5*omega*np.where(y>=0,large,small)
    qm=0.5*omega*np.where(y>=0,small,large)
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
    shape=tuple(shape)
    if not shape or any(not isinstance(v,(int,np.integer)) or v<=0 for v in shape): raise ValueError('shape must contain positive integer dimensions.')
    if not np.isfinite(cross_weight) or cross_weight<=0 or not np.isfinite(between_block_weight) or between_block_weight<=0: raise ValueError('graph weights must be finite and positive.')
    N=int(np.prod(shape)); base=[]
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
    d=finite_array(d,name='observed').ravel(); rho,_=lift(d,eps)
    if data_shape is None: data_shape=(d.size,)
    data_shape=acquisition_shape(data_shape,d.size)
    edges=graph_edges(two_layer_graph_edges(data_shape,cross_weight,between_block_weight),2*d.size)
    B=B_from_rho(rho,edges)
    solver=spla.factorized(B[:-1,:-1].tocsc())
    return {'d':d,'rho_d':rho,'B':B,'edges':edges,'eps':eps,'shape':data_shape,'solver':solver}


def frozen_objective_grad(y, setup):
    y=finite_array(y,name='predicted').ravel()
    if y.size != np.asarray(setup['d']).size: raise ValueError('prediction size must match the fitted observation.')
    rho,_=lift(y,setup['eps'])
    dr=rho-setup['rho_d']; b=dr-dr.mean(); xr=setup['solver'](b[:-1]); eta=np.r_[xr,0.0]; eta-=eta.mean()
    val=0.5*float(dr@eta)
    gy=jt_eta(y,eta,setup['eps'])
    return val,gy


def normalized_l2_objective_grad(y,d):
    y=finite_array(y,name='predicted').ravel(); d=finite_array(d,name='observed').ravel()
    if y.size != d.size: raise ValueError('prediction size must match the observation.')
    ny=np.linalg.norm(y); nd=np.linalg.norm(d)
    if ny==0 or nd==0: raise ValueError('Normalized-L2 requires nonzero prediction and observation vectors.')
    uy=y/ny; ud=d/nd; r=uy-ud
    val=0.5*r@r
    gy=(r - uy*(uy@r))/ny
    return float(val),gy


def l2_objective_grad(y,d):
    y=finite_array(y,name='predicted').ravel(); d=finite_array(d,name='observed').ravel()
    if y.size != d.size: raise ValueError('prediction size must match the observation.')
    r=y-d
    scale=max(np.linalg.norm(d)**2,1e-30)
    return 0.5*float(r@r)/scale, r/scale



def fixed_scale_lift(y, reference_scale2, eps=0.15, omega=None):
    y=finite_array(y,name='signal').ravel(); n=y.size
    if not np.isfinite(eps) or eps<=0: raise ValueError('eps must be finite and positive.')
    reference_scale2=float(reference_scale2)
    if not np.isfinite(reference_scale2) or reference_scale2<=0: raise ValueError('reference_scale2 must be finite and positive.')
    omega=positive_weights(omega,n)
    r=np.sqrt(y*y + eps*eps*reference_scale2)
    Z=(omega*r).sum()
    # Rationalization avoids cancellation in the smaller species when eps is small.
    large=r+np.abs(y)
    small=(eps*eps*float(reference_scale2))/large
    qp=0.5*omega*np.where(y>=0,large,small)
    qm=0.5*omega*np.where(y>=0,small,large)
    rho=np.r_[qp,qm]/Z
    return rho, (r,Z,omega)


def fixed_scale_jt_eta(y, eta, reference_scale2, eps=0.15, omega=None):
    y=finite_array(y,name='signal').ravel(); n=y.size
    eta=finite_array(eta,name='eta').ravel()
    if eta.size != 2*n: raise ValueError('eta must have exactly twice the signal length.')
    omega=positive_weights(omega,n)
    rho,(r,Z,omega)=fixed_scale_lift(y,reference_scale2,eps,omega)
    ep=eta[:n]; em=eta[n:]
    cp=ep@rho[:n] + em@rho[n:]
    wt=omega*(ep+em-2*cp)/(2*Z)
    term1=omega*(ep-em)/(2*Z)
    term2=wt*(y/r)
    return term1+term2


def fixed_scale_setup(d, eps=0.15, data_shape=None, cross_weight=0.25, between_block_weight=0.2):
    d=finite_array(d,name='observed').ravel()
    if not np.any(d!=0): raise ValueError('fixed-scale setup requires a nonzero observation.')
    reference_scale2=float(np.mean(d*d))
    rho,_=fixed_scale_lift(d,reference_scale2,eps)
    if data_shape is None: data_shape=(d.size,)
    data_shape=acquisition_shape(data_shape,d.size)
    edges=graph_edges(two_layer_graph_edges(data_shape,cross_weight,between_block_weight),2*d.size)
    B=B_from_rho(rho,edges)
    solver=spla.factorized(B[:-1,:-1].tocsc())
    return {'d':d,'rho_d':rho,'B':B,'edges':edges,'eps':eps,'shape':data_shape,
            'solver':solver,'reference_scale2':reference_scale2}


def fixed_scale_objective_grad(y, setup):
    y=finite_array(y,name='predicted').ravel()
    if y.size != np.asarray(setup['d']).size: raise ValueError('prediction size must match the fitted observation.')
    rho,_=fixed_scale_lift(y,setup['reference_scale2'],setup['eps'])
    dr=rho-setup['rho_d']; b=dr-dr.mean()
    xr=setup['solver'](b[:-1]); eta=np.r_[xr,0.0]; eta-=eta.mean()
    val=0.5*float(dr@eta)
    gy=fixed_scale_jt_eta(y,eta,setup['reference_scale2'],setup['eps'])
    return val,gy
