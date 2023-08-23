import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from kmodes.kmodes import KModes

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from estyp.cluster import NClusterSearch


def test_nclustersearch_initialization():
    searcher = NClusterSearch(estimator=KMeans(n_init="auto"), method='elbow')
    assert searcher.method == 'elbow'
    assert searcher.min_clusters == 1
    assert searcher.max_clusters == 10


def test_nclustersearch_elbow_method():
    data = load_iris().data
    searcher = NClusterSearch(estimator=KMeans(n_init="auto"), method='elbow')
    searcher.fit(data)
    assert isinstance(searcher.optimal_clusters_, int)


def test_nclustersearch_silhouette_method():
    data = load_iris().data
    searcher = NClusterSearch(estimator=KMeans(
        n_init="auto"), method='silhouette', min_clusters=2)
    searcher.fit(data)
    assert isinstance(searcher.optimal_clusters_, int)


def test_nclustersearch_predict():
    data = load_iris().data
    searcher = NClusterSearch(estimator=KMeans(n_init="auto"), method='elbow')
    searcher.fit(data)
    predictions = searcher.predict(data[:10])
    assert len(predictions) == 10
    assert np.all(predictions >= 0)


def test_nclustersearch_invalid_cluster_numbers():
    with pytest.raises(Exception):
        NClusterSearch(estimator=KMeans(n_init="auto"),
                       min_clusters=10, max_clusters=5)


def test_nclustersearch_fit_with_invalid_sample_size():
    data = np.array([[1, 2], [3, 4]])
    searcher = NClusterSearch(estimator=KMeans(n_init="auto"), max_clusters=3)
    with pytest.raises(Exception):
        searcher.fit(data)


def test_nclustersearch_invalid_method():
    with pytest.raises(Exception):
        NClusterSearch(estimator=KMeans(n_init="auto"),
                       method='unknown_method')


def test_nclustersearch_predict_without_fit():
    data = load_iris().data
    searcher = NClusterSearch(estimator=KMeans(n_init="auto"), method='elbow')
    with pytest.raises(Exception):
        searcher.predict(data[:10])


def test_nclustersearch_labels():
    data = load_iris().data
    searcher = NClusterSearch(estimator=KMeans(n_init="auto"), method='elbow')
    searcher.fit(data)
    assert len(searcher.labels_) == len(data)


def test_nclustersearch_repr():
    searcher = NClusterSearch(estimator=KMeans(n_init="auto"), method='elbow')
    repr_str = repr(searcher)
    assert isinstance(repr_str, str)
    assert "NClusterSearch(estimator=" in repr_str


def test_nclustersearch_with_kmodes():
    np.random.seed(123)
    data = np.random.randint(0, 10, size=(100, 4))
    kmodes = KModes(init='Huang', n_init=5)
    searcher = NClusterSearch(estimator=kmodes, method='elbow')
    searcher.fit(data)
    assert searcher.optimal_clusters_ in range(1, 11)


def test_nclustersearch_with_pipeline():
    np.random.seed(123)
    data = np.random.randint(0, 10, size=(100, 4))
    kmodes = KModes(init='Huang', n_init=5)
    pipeline = make_pipeline(StandardScaler(), kmodes)
    searcher = NClusterSearch(estimator=pipeline, method='elbow')
    searcher.fit(data)
    assert searcher.optimal_clusters_ in range(1, 11)


def test_nclustsearch_parallel():
    np.random.seed(123)
    data = np.random.randint(0, 10, size=(100, 4))
    searcher = NClusterSearch(estimator=KMeans(
        n_init="auto"), method='elbow', n_jobs=2)
    searcher.fit(data)
    assert searcher.optimal_clusters_ in range(1, 11)
