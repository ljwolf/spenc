import scipy.sparse.csgraph as csg
import scipy.sparse as sp
from warnings import warn as Warn
import numpy as np

def check_weights(W, X=None, transform = None):
    """
    Check that the provided weights matrix and the X matrix are conformal.
    Further, check that the spatial weights are fully connected. 
    """
    if X is not None:
        assert W.shape[0] == X.shape[0], "W does not have the same number of samples as X"
    graph = sp.csc_matrix(W)
    graph.eliminate_zeros()
    components, labels = csg.connected_components(graph)
    if components > 1:
    	Warn('Spatial affinity matrix is disconnected, and has {} subcomponents.'
    		 'This will certainly affect the solution output.')
    return W

def lattice(x,y):
    """
    Construct a lattice of unit squares of dimension (x,y)
    """
    from shapely.geometry import Polygon
    import geopandas as gpd
    x = np.arange(x)*1.0
    y = np.arange(y)*1.0
    pgons = []
    for i in x:
        for j in y:
            ll,lr,ur,ul = (i,j), (i+1,j),\
                          (i+1,j+1), (i,j+1)
            #print([ll,lr,ur,ul])
            pgons.append(Polygon([ll,lr,ur,ul]))
    return gpd.GeoDataFrame({'geometry':pgons})

def p_connected(replications):
    """
    Compute the probability that any two observations are clustered
    together through a set of labellings.

    Uses outer product broadcasting in numpy, so only iterates over n_iterations,
    rather than n_iterations X n_iterations.
    """
    n_replications, n_observations = replications.shape
    # dumbest way to do this
    out = np.zeros((n_observations, n_observations))
    for replication in replications:
        out += replication[:,None] == replication[None,:]
    return out/len(replications)