"""Input-domain validation used by the public and low-level objectives."""
import numpy as np


def finite_array(signal, *, name='signal', min_size=1):
    a=np.asarray(signal,dtype=float)
    if a.size < int(min_size) or not np.isfinite(a).all():
        raise ValueError(f'{name} must contain at least {int(min_size)} finite sample(s).')
    return a


def positive_weights(omega, size):
    if omega is None:
        return np.ones(int(size), dtype=float)
    w=np.asarray(omega,dtype=float).ravel()
    if w.size != int(size) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('omega must have one finite positive weight per signal sample.')
    return w


def coordinate_axis(axis, size, *, name='coordinate axis'):
    x=np.asarray(axis,dtype=float)
    if x.ndim != 1 or x.size != int(size) or not np.isfinite(x).all():
        raise ValueError(f'{name} must be a finite 1D array matching the trace length.')
    return x


def signal_vector(signal, epsilon):
    a=finite_array(signal,name='signal',min_size=2)
    if not np.any(a!=0):
        raise ValueError('Expected at least two finite samples and a nonzero signal.')
    if not np.isfinite(epsilon) or epsilon<=0:
        raise ValueError('epsilon must be finite and positive.')
    return a


def acquisition_shape(shape,size):
    shape=tuple(shape)
    if not shape or any(not isinstance(x,(int,np.integer)) or x<=0 for x in shape) or np.prod(shape)!=size:
        raise ValueError('data_shape must contain positive integer dimensions matching signal size.')
    return shape


def graph_edges(edges,node_count):
    edges=list(edges);parent=list(range(node_count))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for edge in edges:
        if len(edge)!=3:raise ValueError('Each edge must be (i,j,positive conductance).')
        i,j,w=edge
        if not isinstance(i,(int,np.integer)) or not isinstance(j,(int,np.integer)) or not(0<=i<node_count and 0<=j<node_count) or i==j:
            raise ValueError('Edge endpoints must be distinct valid integer node indices.')
        if not np.isfinite(w) or w<=0:raise ValueError('Edge conductances must be finite and positive.')
        parent[find(i)]=find(j)
    if len({find(i) for i in range(node_count)})!=1:raise ValueError('The acquisition graph must be connected.')
    return edges
