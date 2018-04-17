from sklearn import cluster as clust
import sklearn.metrics as skm
import sklearn.metrics.pairwise as pw
from sklearn.utils.validation import check_array
from .utils import check_weights
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.extmath import _deterministic_vector_sign_flip
from sklearn.utils import check_random_state
from sklearn.cluster.spectral import discretize as _discretize
import numpy as np
from .scores import boundary_fraction
from scipy.sparse import csgraph as cg, linalg as la

class SPENC(clust.SpectralClustering):

    def fit(self, X, W=None, y=None, shift_invert=True, breakme=False, check_W=True):
        """Creates an affinity matrix for X using the selected affinity,
        applies W to the affinity elementwise, and then applies spectral clustering
        to the affinity matrix.
        
        NOTE:

        breakme sends the affinity matrix down to scikit's spectral clustering class.
        I call this breakme because of bug8129. 
        I don't see a significant difference here when switching between the two, 
        most assignments in the problems I've examined are the same. I think, since the bug is in the scaling of the eigenvectors, it's not super important. 
        """
        if X is not None:
            X = check_array(X, accept_sparse = ['csr','coo', 'csc'],
                        dtype=np.float64, ensure_min_samples=2)
            if check_W:
                W = check_weights(W, X)
            
            if self.affinity == 'nearest_neighbors':
                connectivity = kneighbors_graph(X, n_neighbors=self.n_neighbors,
                                                include_self=True, n_jobs=self.n_jobs)
                self.affinity_matrix_ = .5 * (connectivity + connectivity.T)
            elif self.affinity == 'precomputed':
                self.affinity_matrix_ = X
            else:
                params = self.kernel_params
                if params is None:
                    params = {}
                if not callable(self.affinity):
                    params['gamma'] = self.gamma
                    params['degree'] = self.degree
                    params['coef0'] = self.coef0
                self.attribute_affinity_ = pw.pairwise_kernels(X, metric=self.affinity,
                                                               filter_params=True,
                                                               **params)
                self.spatial_affinity_ = W
                self.affinity_matrix_ = W.multiply(self.attribute_affinity_)
        else:
            self.affinity_matrix_ = W
        if breakme:
            self.affinity_ = self.affinity
            self.affinity = 'precomputed'
            super().fit(self.affinity_matrix_, y=y)
        
            self.affinity = self.affinity_
            del self.affinity_
            return self

        laplacian, orig_d = cg.laplacian(self.affinity_matrix_, 
                                          normed=True, return_diag=True)
        laplacian *=-1

        random_state = check_random_state(self.random_state)
        v0 = random_state.uniform(-1,1,laplacian.shape[0])
        
        if not shift_invert:
            ev, spectrum = la.eigsh(laplacian, which='LA', k=self.n_clusters, v0=v0)
        else:
            ev, spectrum = la.eigsh(laplacian, which='LM', sigma=1, k=self.n_clusters, v0=v0)
        embedding = spectrum.T[self.n_clusters::-1] #sklearn/issues/8129
        embedding = embedding / orig_d
        embedding = _deterministic_vector_sign_flip(embedding)
        self.embedding_ = embedding
        if self.assign_labels == 'kmeans':
            self.labels_ = clust.KMeans(n_clusters=self.n_clusters).fit(embedding.T).labels_
        else:
            self.labels_ = _discretize(embedding.T, random_state=random_state)
        return self


    def score(self, X, W, labels=None, delta=.5, 
              attribute_score=skm.calinski_harabaz_score,
              spatial_score=boundary_fraction,
              attribute_kw = dict(),
              spatial_kw = dict()
              ):
        """
        Computes the score of the given label vector on data in X using convex
        combination weight in delta. 

        Arguments
        ---------
        X               : numpy array (N,P)
                          array of data classified into `labels` to score.
        W               : sparse array or numpy array (N,N)
                          array representation of spatial relationships
        labels          : numpy array (N,)
                          vector of labels aligned with X and W
        delta           : float
                          weight to apply to the attribute score. 
                          Spatial score is given weight 1 - delta, 
                          and attributes weight delta. 
                          Default: .5
        attribute_score : callable
                          function to use to evaluate attribute homogeneity
                          Must have signature attribute_score(X,labels,**params)
                          Default: sklearn.metrics.calinski_harabaz_score 
                                   (within/between deviation ratio)
        spatial_score   : callable
                          function to use to evaluate spatial regularity/contiguity.
                          Must have signature spatial_score(X,labels,**params)
                          Default: boundary_ratio(W,X,labels,**spatial_kw)
        """
        if labels is None:
            if not hasattr(self, 'labels_'):
                raise Exception('Object must be fit in order to avoid passing labels.')
            labels = self.labels_
        labels = np.asarray(labels).flatten()
        attribute_score = attribute_score(X,labels, **attribute_kw)
        spatial_score = spatial_score(W,labels, X=X,**spatial_kw)
        return delta * attribute_score + (1 - delta)*spatial_score
