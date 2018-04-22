from unittest import TestCase
import pysal as ps
import numpy as np
import os
import geopandas as gpd
import scipy.sparse.csgraph as csg

from spenc.utils import lattice
import spenc 

filepath = os.path.dirname(__file__)


class SPENCTest(TestCase):
    def setUp(self):
        self.nat = gpd.read_file(ps.examples.get_path('NAT.shp'))
        self.natR = ps.weights.Rook.from_dataframe(self.nat)
        self.nat_10k_nodata = np.load(os.path.join(filepath, 'data/nat_10k_nodata.ary'))
        self.nat_30k_randoms = np.load(os.path.join(filepath, 'data/nat_30k_randoms.ary'))
        self.nat_30k_discovered = np.load(os.path.join(filepath, 'data/nat_30k_discovered.ary'))
        self.nat_infk_randoms = np.load(os.path.join(filepath, 'data/nat_infk_randoms.ary'))
        self.nat_infk_discovered = np.load(os.path.join(filepath, 'data/nat_infk_discovered.ary'))
        self.nat_names = self.nat.filter(like='90').columns.tolist() \
                  + self.nat.filter(like='89').columns.tolist()
        self.natX = self.nat[self.nat_names].values
        self.natX = (self.natX - self.natX.mean(axis=0)) / self.natX.var(axis=0)

    def test_NAT_nodata(self):
        np.random.seed(1901) #shouldn't matter substantively, only for label#
        t1 = spenc.SPENC(n_clusters=10).fit(None, self.natR.sparse).labels_
        for label in range(t1.max()):
            mask = t1 == label
            subgraph = self.natR.sparse[mask,:][:,mask]
            subgraph.eliminate_zeros()
            n_components, labels = csg.connected_components(subgraph)
            self.assertEqual(n_components, 1,
                             'Disconnected component ({}) in NAT clusters!'.format(label))
        np.testing.assert_array_equal(t1, self.nat_10k_nodata)

    def test_NAT_randoms(self):
        np.random.seed(1901)
        randoms = spenc.SPENC(n_clusters=30).sample(self.natR.sparse, n_samples=3)
        self.assertEqual(randoms.shape, (3, len(self.nat)), 'sample shapes are incorrect!')
        for i,random in enumerate(randoms):
            for label in range(random.max()):
                mask = random == label
                subgraph = self.natR.sparse[mask,:][:,mask]
                subgraph.eliminate_zeros()
                n_components, labels = csg.connected_components(subgraph)
                self.assertEqual(n_components, 1,
                                 'Disconnected component ({}) in NAT '
                                 'random cluster set {}!'.format(label,i) )
        np.testing.assert_array_equal(randoms, self.nat_30k_randoms)
        np.random.seed(1901)
        randoms = spenc.SPENC(n_clusters=np.inf).sample(self.natR.sparse, floor=20)
        self.assertEqual(randoms.shape, (len(self.nat),), 'sample shapes are incorrect!')
        for label in range(randoms.max()):
            mask = randoms == label
            subgraph = self.natR.sparse[mask,:][:,mask]
            subgraph.eliminate_zeros()
            n_components, labels = csg.connected_components(subgraph)
            self.assertEqual(n_components, 1,
                             'Disconnected component ({}) in NAT '
                             'random cluster set {}!'.format(label,i))
        np.testing.assert_array_equal(randoms, self.nat_infk_randoms)

    def test_NAT_data(self):
        np.random.seed(1901)
        k30 = spenc.SPENC(n_clusters=30, gamma=.001).fit(self.natX, self.natR.sparse)
        for label in range(k30.labels_.max()):
            mask = k30.labels_ == label
            subgraph = self.natR.sparse[mask,:][:,mask]
            subgraph.eliminate_zeros()
            n_components, labels = csg.connected_components(subgraph)
            self.assertEqual(n_components, 1,
                             'Disconnected component ({}) in NAT clusters!'.format(label))
        np.testing.assert_array_equal(k30.labels_, self.nat_30k_discovered)
        np.random.seed(1901)
        kinf = spenc.SPENC(n_clusters=np.inf, gamma=.001)\
                    .fit(self.natX, self.natR.sparse, floor=20)
        for label in range(kinf.labels_.max()):
            mask = kinf.labels_ == label
            subgraph = self.natR.sparse[mask,:][:,mask]
            subgraph.eliminate_zeros()
            n_components, labels = csg.connected_components(subgraph)
            self.assertEqual(n_components, 1,
                             'Disconnected component ({}) in NAT clusters!'.format(label))
        np.testing.assert_array_equal(kinf.labels_, self.nat_infk_discovered)
