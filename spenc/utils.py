import scipy.sparse.csgraph as csg
import scipy.sparse as sp
from warnings import warn as Warn

def check_weights(W, X=None, transform = None):
    if X is not None:
        assert W.shape[0] == X.shape[0], "W does not have the same number of samples as X"
    graph = sp.csc_matrix(W)
    graph.eliminate_zeros()
    components, labels = csg.connected_components(graph)
    if components > 1:
    	Warn('Spatial affinity matrix is disconnected, and has {} subcomponents.'
    		 'This will certainly affect the solution output.')
    return W
