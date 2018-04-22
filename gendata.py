import numpy as np
import spenc as spenc
import pysal as ps
import geopandas as gpd
import os
SEED = 1901
dir_self = os.path.dirname(__file__)
datadir = os.path.join(dir_self, 'spenc/tests/data')

if __name__ == "__main__":
    nat = gpd.read_file(ps.examples.get_path("NAT.shp"))
    natR = ps.weights.Rook.from_dataframe(nat)
    names = nat.filter(like='90').columns.tolist() + nat.filter(like='89').columns.tolist()
    X = nat[names].values
    X = (X - X.mean(axis=0))/X.var(axis=0)

    print('(1 of 5) doing 10k nodata')
    np.random.seed(SEED)
    labels = spenc.SPENC(n_clusters=10, random_state=SEED).fit(None, natR.sparse).labels_
    labels.dump(os.path.join(datadir, 'nat_10k_nodata.ary'))

    print('(2 of 5) doing 30k sampling')
    np.random.seed(SEED)
    labels = spenc.SPENC(n_clusters=30, random_state=SEED).sample(natR.sparse, n_samples=3)
    labels.dump(os.path.join(datadir, 'nat_30k_randoms.ary'))

    print('(3 of 5) doing 30k withdata')
    np.random.seed(SEED)
    labels = spenc.SPENC(n_clusters=30, gamma=.001, random_state=SEED).fit(X, natR.sparse).labels_
    labels.dump(os.path.join(datadir, 'nat_30k_discovered.ary'))

    print('(4 of 5) doing infk sampling')
    np.random.seed(SEED)
    labels = spenc.SPENC(n_clusters=np.inf, random_state=SEED).sample(natR.sparse, floor=20)
    labels.dump(os.path.join(datadir, 'nat_infk_randoms.ary'))

    print('(5 of 5) doing infk withdata')
    np.random.seed(SEED)
    labels = spenc.SPENC(n_clusters=np.inf, gamma=.001, random_state=SEED).fit(X, natR.sparse, floor=20).labels_
    labels.dump(os.path.join(datadir, 'nat_infk_discovered.ary'))


    print('done!')

